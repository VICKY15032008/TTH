import os
import datetime
import shutil
import subprocess
import time
import traceback
import struct
import tempfile
from collections import Counter
from flask import Flask, request, render_template_string, send_file
import fitz
import hashlib
import marisa_trie
import Levenshtein
from tqdm import tqdm
from PIL import Image
import cProfile
from pstats import Stats

app = Flask(__name__)

UPLOAD_HTML = """
<!DOCTYPE html>
<html>
<head><title>PDF Compression & Reconstruction</title></head>
<body>
<h1>Upload PDF</h1>
<form method="POST" enctype="multipart/form-data">
  <input type="file" name="pdf_file" accept="application/pdf" required><br><br>
  <label>Select output color mode:</label>
  <select name="color_mode">
    <option value="rgb">RGB</option>
    <option value="gray">Grayscale</option>
  </select><br><br>
  <button type="submit">Process</button>
</form>
{% if error %}<p style="color:red;">{{ error }}</p>{% endif %}
</body></html>
"""

RESULT_HTML = """
<!DOCTYPE html>
<html><head><title>Results</title></head><body>
<h1>Results</h1>
<p>Text Similarity: {{ (text_similarity * 100) | round(2) }}%</p>
<p>Trie Text Fidelity: {{ (trie_text_fidelity * 100) | round(2) }}%</p>
<p>Image Similarity: {{ (image_similarity * 100) | round(2) }}%</p>
<p>Compression Ratio (Reconstructed/Input): {{ (compression_ratio * 100) | round(2) }}%</p>
<p>Original PDF size: {{ orig_size }} MB</p>
<p>Compressed PDF size: {{ compressed_size }} MB</p>
<p>Reconstructed PDF size: {{ reconstructed_size }} MB</p>
<p>Compressed Text (Trie) size: {{ trie_file_size }} MB</p>
<p>Headers count: {{ headers_count }} | Total header size: {{ headers_size }} MB</p>
<p>Total reconstruction overhead: {{ reconstruction_overhead }} MB</p>
<p><a href="/download?path={{ compressed_pdf }}">Download Compressed PDF</a></p>
<p><a href="/download?path={{ reconstructed_pdf }}">Download Reconstructed PDF</a></p>
<a href="/">Process another file</a>
</body></html>
"""

def file_size_mb(path):
    try:
        return os.path.getsize(path) / (1024 * 1024)
    except Exception:
        return 0

def extract_header_bytes(page, height=80):
    try:
        pix = page.get_pixmap(clip=fitz.Rect(0, 0, page.rect.width, height))
        return pix.tobytes()
    except Exception:
        return b""

def run_ocrmypdf(INPUT_PDF, COMPRESSED_PDF, errors_encountered, timings):
    start = time.time()
    strategies = [
        ["ocrmypdf", "--optimize", "1", "--skip-text", "--color-conversion-strategy=RGB", INPUT_PDF, COMPRESSED_PDF],
        ["ocrmypdf", "--optimize", "1", "--skip-text", "--output-type", "pdf", INPUT_PDF, COMPRESSED_PDF],
        ["ocrmypdf", "--skip-text", INPUT_PDF, COMPRESSED_PDF],
    ]
    for idx, cmd in enumerate(strategies):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode == 0:
                elapsed = time.time() - start
                timings['ocrmypdf_compression'] = elapsed
                return
            else:
                errors_encountered.append(f"Strategy {idx+1} failed: {result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            errors_encountered.append(f"Strategy {idx+1} timeout (>600s)")
        except Exception as e:
            errors_encountered.append(f"Strategy {idx+1} exception: {e}")
    try:
        shutil.copy(INPUT_PDF, COMPRESSED_PDF)
        timings['ocrmypdf_compression'] = time.time() - start
    except Exception as e:
        raise RuntimeError(f"Cannot copy PDF after OCRmyPDF failure: {e}")

def extract_headers_from_original(INPUT_PDF, HEADER_DIR, errors_encountered, timings):
    start = time.time()
    header_hashes = set()
    doc = fitz.open(INPUT_PDF)
    for i in tqdm(range(len(doc)), desc="Header extraction", leave=False):
        try:
            page = doc[i]
            header_data = extract_header_bytes(page)
            if header_data:
                h_hash = hashlib.md5(header_data).hexdigest()
                if h_hash not in header_hashes:
                    header_hashes.add(h_hash)
                    header_path = os.path.join(HEADER_DIR, f'header_{h_hash}.bin')
                    try:
                        with open(header_path, 'wb') as f:
                            f.write(header_data)
                    except Exception as e:
                        errors_encountered.append(f"Header save error page {i}: {e}")
        except Exception as e:
            errors_encountered.append(f"Page {i} processing error: {e}")
    doc.close()
    timings['header_extraction'] = time.time() - start

def extract_tokens_from_compressed(COMPRESSED_PDF, TRIE_FILE, TOKEN_DIR, errors_encountered, timings):
    start = time.time()
    doc = fitz.open(COMPRESSED_PDF)
    all_page_tokens = []
    global_tokens = set()
    for i in tqdm(range(len(doc)), desc="Tokens extraction", leave=False):
        try:
            page = doc[i]
            try:
                text = page.get_text().strip()
                if text:
                    page_tokens = text.split()
                    all_page_tokens.append(page_tokens)
                    global_tokens.update(page_tokens)
            except Exception as e:
                errors_encountered.append(f"Text extraction error page {i}: {e}")
                all_page_tokens.append([])
        except Exception as e:
            errors_encountered.append(f"Page {i} processing error: {e}")
            all_page_tokens.append([])
    doc.close()
    try:
        if global_tokens:
            unique_sorted = sorted(global_tokens)
            trie = marisa_trie.Trie(unique_sorted)
            trie.save(TRIE_FILE)
            globals()['unique_tokens'] = unique_sorted
        else:
            globals()['unique_tokens'] = []
    except Exception as e:
        errors_encountered.append(f"Trie save error: {e}")
    timings['token_extraction'] = time.time() - start

def encode_to_base64_streaming(COMPRESSED_PDF, COMPRESSED_PDF_BASE64, errors_encountered, timings):
    start = time.time()
    try:
        with open(COMPRESSED_PDF, 'rb') as pdf_file, open(COMPRESSED_PDF_BASE64, 'w') as b64_file:
            while True:
                chunk = pdf_file.read(8192)
                if not chunk:
                    break
                b64_file.write(base64.b64encode(chunk).decode())
        timings['base64_encoding'] = time.time() - start
    except Exception as e:
        errors_encountered.append(f"Base64 encoding error: {e}")
        timings['base64_encoding'] = time.time() - start

def simulate_reconstruction_streaming(COMPRESSED_PDF, RECONSTRUCTED_PDF, errors_encountered, timings):
    start = time.time()
    try:
        with open(COMPRESSED_PDF, 'rb') as src, open(RECONSTRUCTED_PDF, 'wb') as dst:
            while True:
                chunk = src.read(8192)
                if not chunk:
                    break
                dst.write(chunk)
        timings['reconstruction_sim'] = time.time() - start
    except Exception as e:
        errors_encountered.append(f"Reconstruction error: {e}")
        timings['reconstruction_sim'] = time.time() - start

def convert_reconstructed_to_grayscale(RECONSTRUCTED_PDF, errors_encountered):
    try:
        if not os.path.exists(RECONSTRUCTED_PDF):
            errors_encountered.append("Reconstructed PDF not found for grayscale conversion")
            return
        doc = fitz.open(RECONSTRUCTED_PDF)
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
        os.close(tmp_fd)
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(colorspace=fitz.csRGB)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            gray_img = img.convert('L')
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_png:
                gray_img.save(tmp_png.name, format='PNG')
                tmp_png_path = tmp_png.name
            gray_pix = fitz.Pixmap(tmp_png_path)
            page.clean_contents()
            rect = fitz.Rect(0, 0, gray_pix.width, gray_pix.height)
            page.insert_image(rect, pixmap=gray_pix)
            gray_pix = None
            pix = None
            os.remove(tmp_png_path)
        doc.save(tmp_path, garbage=4, deflate=True, clean=True, incremental=False)
        doc.close()
        os.replace(tmp_path, RECONSTRUCTED_PDF)
    except Exception as e:
        errors_encountered.append(f"Grayscale conversion error: {e}")

def extract_text_and_images_streaming(doc_path, errors_encountered):
    if not os.path.exists(doc_path):
        return "", Counter()
    try:
        doc = fitz.open(doc_path)
        text_parts = []
        image_hashes = Counter()
        for page_num in range(len(doc)):
            try:
                page = doc[page_num]
                try:
                    txt = page.get_text()
                    text_parts.append(txt)
                except:
                    pass
                try:
                    for img in page.get_images(full=True):
                        xref = img[0]
                        try:
                            base_img = doc.extract_image(xref)
                            img_bytes = base_img["image"]
                            img_hash = hashlib.sha256(img_bytes).hexdigest()
                            image_hashes[img_hash] += 1
                        except:
                            pass
                except:
                    pass
            except:
                continue
        doc.close()
        full_text = ''.join(text_parts)
        text_parts.clear()
        return full_text, image_hashes
    except Exception as e:
        errors_encountered.append(f"Extract error for {doc_path}: {e}")
        return "", Counter()

def levenshtein_similarity(text1, text2, errors_encountered):
    if not text1 or not text2:
        return 0.0
    try:
        MAX_CHARS = 100000
        if len(text1) > MAX_CHARS or len(text2) > MAX_CHARS:
            text1 = text1[:MAX_CHARS]
            text2 = text2[:MAX_CHARS]
        return Levenshtein.ratio(text1, text2)
    except Exception as e:
        errors_encountered.append(f"Levenshtein error: {e}")
        return 0.0

def image_overlap(hash1, hash2, errors_encountered):
    try:
        common = set(hash1) & set(hash2)
        common_count = sum(min(hash1[h], hash2[h]) for h in common)
        total = sum(hash1.values())
        return common_count / total if total else 0
    except Exception as e:
        errors_encountered.append(f"Image overlap error: {e}")
        return 0.0

def compress_text_with_trie(COMPRESSED_PDF, TRIE_FILE, COMPRESSED_TOKENS, TOKEN_OFFSETS, errors_encountered, timings):
    start = time.time()
    if not os.path.exists(COMPRESSED_PDF) or 'unique_tokens' not in globals():
        errors_encountered.append("Missing compressed PDF or trie for text compression")
        return 0, 0
    orig_text_bytes = 0
    compressed_data = b''
    offsets = []
    orig_texts = []
    offset = 0
    try:
        doc = fitz.open(COMPRESSED_PDF)
        token_to_id = {token: idx for idx, token in enumerate(globals()['unique_tokens'])}
        num_pages = len(doc)
        for page_num in tqdm(range(num_pages), desc="Compressing page texts", leave=False):
            try:
                page = doc[page_num]
                text = page.get_text().strip()
                page_orig_bytes = len(text.encode('utf-8'))
                orig_text_bytes += page_orig_bytes
                orig_texts.append(text)
                page_tokens = text.split()
                page_compressed = b''
                for token in page_tokens:
                    if token in token_to_id:
                        page_compressed += struct.pack('>H', token_to_id[token])
                    else:
                        page_compressed += token.encode('utf-8') + b'\x00'
                compressed_data += page_compressed
                offsets.append(offset)
                offset += len(page_compressed)
            except Exception as e:
                errors_encountered.append(f"Page {page_num} compression error: {e}")
                orig_texts.append('')
                offsets.append(offset)
        doc.close()
        with open(COMPRESSED_TOKENS, 'wb') as f:
            f.write(compressed_data)
        with open(TOKEN_OFFSETS, 'w') as f:
            f.write('\n'.join(map(str, offsets)))
        globals()['orig_compressed_texts'] = orig_texts
        compressed_bytes = len(compressed_data)
        timings['text_trie_compression'] = time.time() - start
        return orig_text_bytes, compressed_bytes
    except Exception as e:
        errors_encountered.append(f"Text trie compression error: {e}")
        timings['text_trie_compression'] = time.time() - start
        return 0, 0

def reconstruct_text_from_trie(COMPRESSED_TOKENS, TOKEN_OFFSETS, TRIE_FILE, errors_encountered, timings):
    start = time.time()
    if not all([os.path.exists(COMPRESSED_TOKENS), os.path.exists(TOKEN_OFFSETS), os.path.exists(TRIE_FILE)]):
        errors_encountered.append("Missing files for text reconstruction")
        return "", 0.0
    if 'orig_compressed_texts' not in globals():
        errors_encountered.append("Missing original compressed texts for fidelity")
        return "", 0.0
    try:
        unique_tokens = globals()['unique_tokens']
        orig_texts = globals()['orig_compressed_texts']
        recon_texts = []
        with open(COMPRESSED_TOKENS, 'rb') as f:
            data = f.read()
        offsets_str = open(TOKEN_OFFSETS).readlines()
        offsets = [int(line.strip()) for line in offsets_str if line.strip()]
        num_pages = len(offsets)
        for i in range(num_pages):
            start_off = offsets[i] if i < len(offsets) else 0
            end_off = offsets[i+1] if i+1 < len(offsets) else len(data)
            page_data = data[start_off:end_off]
            page_tokens = []
            pos = 0
            while pos < len(page_data):
                if pos + 2 <= len(page_data):
                    id_candidate = struct.unpack('>H', page_data[pos:pos+2])[0]
                    if 0 <= id_candidate < len(unique_tokens):
                        page_tokens.append(unique_tokens[id_candidate])
                        pos += 2
                        continue
                    else:
                        pos += 2
                null_pos = page_data.find(b'\x00', pos)
                if null_pos != -1:
                    page_tokens.append(page_data[pos:null_pos].decode('utf-8'))
                    pos = null_pos + 1
                else:
                    page_tokens.append(page_data[pos:].decode('utf-8', errors='ignore'))
                    pos = len(page_data)
            recon_texts.append(' '.join(page_tokens))
        full_recon = ' '.join(recon_texts)
        orig_full = ' '.join(orig_texts)
        fidelity = Levenshtein.ratio(full_recon, orig_full)
        timings['text_reconstruction'] = time.time() - start
        return full_recon, fidelity
    except Exception as e:
        errors_encountered.append(f"Text reconstruction error: {e}")
        timings['text_reconstruction'] = time.time() - start
        return "", 0.0

def similarity_report(INPUT_PDF, RECONSTRUCTED_PDF, COMPRESSED_TOKENS, TOKEN_OFFSETS, TRIE_FILE, errors_encountered, timings):
    orig_text, orig_imgs = extract_text_and_images_streaming(INPUT_PDF, errors_encountered)
    recon_text, recon_imgs = extract_text_and_images_streaming(RECONSTRUCTED_PDF, errors_encountered)
    text_sim = levenshtein_similarity(orig_text, recon_text, errors_encountered)
    img_sim = image_overlap(orig_imgs, recon_imgs, errors_encountered)
    trie_recon, trie_fid = reconstruct_text_from_trie(COMPRESSED_TOKENS, TOKEN_OFFSETS, TRIE_FILE, errors_encountered, timings)
    timings['similarity_analysis'] = time.time() - timings.get('similarity_analysis', 0)
    return text_sim, img_sim, trie_fid

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('pdf_file')
        color_mode = request.form.get('color_mode', 'rgb')
        if not file or not file.filename.endswith('.pdf'):
            return render_template_string(UPLOAD_HTML, error="Please upload a valid PDF file.")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        filename = os.path.splitext(file.filename)[0]
        BASE_DIR = f"pdf_color_{filename}_{timestamp}"
        IN_DIR = os.path.join(BASE_DIR, "input")
        OUT_DIR = os.path.join(BASE_DIR, "output")
        INT_DIR = os.path.join(BASE_DIR, "intermediate")
        HEADER_DIR = os.path.join(INT_DIR, "headers")
        TOKEN_DIR = os.path.join(INT_DIR, "tokens")
        for d in [IN_DIR, OUT_DIR, INT_DIR, HEADER_DIR, TOKEN_DIR]:
            os.makedirs(d, exist_ok=True)

        INPUT_PDF = os.path.join(IN_DIR, "input.pdf")
        COMPRESSED_PDF = os.path.join(OUT_DIR, "compressed.pdf")
        COMPRESSED_PDF_BASE64 = os.path.join(OUT_DIR, "compressed_base64.txt")
        RECONSTRUCTED_PDF = os.path.join(OUT_DIR, "reconstructed.pdf")
        TRIE_FILE = os.path.join(INT_DIR, "text_tokens.marisa")
        COMPRESSED_TOKENS = os.path.join(TOKEN_DIR, "page_tokens.bin")
        TOKEN_OFFSETS = os.path.join(TOKEN_DIR, "offsets.txt")

        file.save(INPUT_PDF)

        errors_encountered = []
        timings = {}

        # Profiling pipeline as your original function (simplified here)
        profiler = cProfile.Profile()
        profiler.enable()
        try:
            extract_headers_from_original(INPUT_PDF, HEADER_DIR, errors_encountered, timings)
        except Exception:
            traceback.print_exc()
        try:
            run_ocrmypdf(INPUT_PDF, COMPRESSED_PDF, errors_encountered, timings)
        except Exception:
            traceback.print_exc()
        try:
            extract_tokens_from_compressed(COMPRESSED_PDF, TRIE_FILE, TOKEN_DIR, errors_encountered, timings)
        except Exception:
            traceback.print_exc()
        try:
            orig_text_b, comp_text_b = compress_text_with_trie(COMPRESSED_PDF, TRIE_FILE, COMPRESSED_TOKENS, TOKEN_OFFSETS, errors_encountered, timings)
            text_compression_ratio = ((1 - comp_text_b / orig_text_b) * 100) if orig_text_b else 0
        except Exception:
            traceback.print_exc()
            text_compression_ratio = 0
        try:
            encode_to_base64_streaming(COMPRESSED_PDF, COMPRESSED_PDF_BASE64, errors_encountered, timings)
        except Exception:
            traceback.print_exc()
        try:
            simulate_reconstruction_streaming(COMPRESSED_PDF, RECONSTRUCTED_PDF, errors_encountered, timings)
        except Exception:
            traceback.print_exc()
        if color_mode == 'gray':
            try:
                convert_reconstructed_to_grayscale(RECONSTRUCTED_PDF, errors_encountered)
            except Exception:
                traceback.print_exc()
        profiler.disable()
        Stats(profiler).sort_stats('cumtime').print_stats(10)

        # Size calculations
        compress_mb = file_size_mb(COMPRESSED_PDF)
        compress_b64_mb = file_size_mb(COMPRESSED_PDF_BASE64)
        orig_mb = file_size_mb(INPUT_PDF)
        trie_mb = file_size_mb(TRIE_FILE)
        header_files = os.listdir(HEADER_DIR)
        header_sizes = [file_size_mb(os.path.join(HEADER_DIR, hf)) for hf in header_files]
        total_headers_mb = sum(header_sizes)
        text_comp_mb = file_size_mb(COMPRESSED_TOKENS)
        total_recon_overhead = trie_mb + total_headers_mb
        compression_ratio = (file_size_mb(RECONSTRUCTED_PDF) / orig_mb) if orig_mb else 0

        # Similarity analysis
        text_sim, img_sim, trie_fid = similarity_report(INPUT_PDF, RECONSTRUCTED_PDF, COMPRESSED_TOKENS, TOKEN_OFFSETS, TRIE_FILE, errors_encountered, timings)

        return render_template_string(RESULT_HTML,
                                      text_similarity=text_sim,
                                      trie_text_fidelity=trie_fid,
                                      image_similarity=img_sim,
                                      compression_ratio=compression_ratio,
                                      orig_size=round(orig_mb,4),
                                      compressed_size=round(compress_mb,4),
                                      reconstructed_size=round(file_size_mb(RECONSTRUCTED_PDF),4),
                                      trie_file_size=round(trie_mb,4),
                                      headers_count=len(header_files),
                                      headers_size=round(total_headers_mb,4),
                                      reconstruction_overhead=round(total_recon_overhead,4),
                                      compressed_pdf=COMPRESSED_PDF,
                                      reconstructed_pdf=RECONSTRUCTED_PDF)
    return render_template_string(UPLOAD_HTML)

@app.route('/download')
def download_file():
    path = request.args.get('path')
    if path and os.path.exists(path):
        return send_file(path, as_attachment=True)
    return "File not found", 404

if __name__ == '__main__':
    print("Starting Flask...")
    app.run(debug=True, port=5050)
