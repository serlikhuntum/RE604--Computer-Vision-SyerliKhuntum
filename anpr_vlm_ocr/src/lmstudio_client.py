import base64
import io
import re
import requests
from PIL import Image

DEFAULT_PROMPT = (
    "What is the license plate number shown in this image? Respond only with the plate number." 
)

MAX_IMAGE_DIMENSION = 640  # perkecil gambar supaya hemat memori saat inferensi VLM

STOPWORDS = {
    "A", "AN", "THE", "IS", "IT", "IN", "ON", "OF", "OR", "TO", "AT", "BY",
    "AND", "FOR", "HAS", "ARE", "WAS", "BE", "AS", "ID", "NO", "WITH", "READS",
    "READ", "SHOWS", "SHOW", "FROM", "THIS", "THAT",
}


def _sanitize(text: str) -> str:
    """Ganti semua karakter selain huruf/angka/spasi menjadi spasi."""
    return re.sub(r"[^A-Z0-9\s]", " ", text)


def extract_plate_candidate(text: str) -> str:
    tokens = _sanitize(text).split()

    anchor_idx = None
    for i, tok in enumerate(tokens):
        if any(ch.isdigit() for ch in tok):
            anchor_idx = i
            break

    if anchor_idx is None:
        return " ".join(tokens)

    prefix = []
    j = anchor_idx - 1
    while j >= 0 and len(prefix) < 2:
        tok = tokens[j]
        if tok.isalpha() and 1 <= len(tok) <= 3 and tok not in STOPWORDS:
            prefix.insert(0, tok)
            j -= 1
        else:
            break

    suffix = []
    k = anchor_idx + 1
    while k < len(tokens) and len(suffix) < 1:
        tok = tokens[k]
        if tok.isalpha() and 1 <= len(tok) <= 3 and tok not in STOPWORDS:
            suffix.append(tok)
            k += 1
        else:
            break

    parts = prefix + [tokens[anchor_idx]] + suffix
    return " ".join(parts)


def encode_image_to_base64(image_path: str, max_dimension: int = MAX_IMAGE_DIMENSION) -> str:
    """Baca gambar, perkecil jika perlu (menjaga rasio aspek), lalu encode ke base64 JPEG.

    Mengecilkan resolusi gambar sebelum dikirim ke VLM sangat membantu di
    perangkat dengan RAM/VRAM terbatas, karena jumlah "vision token" yang
    harus diproses model berbanding lurus dengan resolusi gambar.
    """
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > max_dimension:
            scale = max_dimension / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=90)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


def query_vlm_plate(
    image_path: str,
    base_url: str = "http://127.0.0.1:1234",
    model: str = "smolvlm-500m-instruct",
    prompt: str = DEFAULT_PROMPT,
    timeout: int = 180,
) -> str:
    """Kirim satu gambar ke LM Studio dan kembalikan teks prediksi plat nomor mentah."""

    b64_image = encode_image_to_base64(image_path)
    image_url = f"data:image/jpeg;base64,{b64_image}"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": 20,
    }

    endpoint = f"{base_url.rstrip('/')}/v1/chat/completions"
    response = requests.post(endpoint, json=payload, timeout=timeout)

    if not response.ok:
        # sertakan isi respons server (biasanya berupa pesan error yang jelas)
        # supaya mudah didiagnosis, bukan cuma kode statusnya saja.
        raise RuntimeError(
            f"LM Studio mengembalikan status {response.status_code}: {response.text}"
        )

    data = response.json()
    raw_text = data["choices"][0]["message"]["content"]
    return raw_text.strip()


def clean_prediction(raw_text: str) -> str:
    """Bersihkan output mentah model menjadi kandidat string plat nomor.

    Menghapus tanda kutip, titik, baris baru, dan spasi berlebih, lalu
    menyeragamkan menjadi huruf besar. Jika model menjawab dengan kalimat
    panjang (mis. "THE LICENSE PLATE IS B 1234 XYZ"), coba ekstrak pola
    yang mirip plat nomor saja dari kalimat itu.
    """
    text = raw_text.strip()
    text = text.replace('"', "").replace("'", "").replace("\n", " ")
    text = text.replace(".", "")
    text = " ".join(text.split())  # rapikan spasi ganda
    text = text.upper()
    text = extract_plate_candidate(text)
    return text


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python lmstudio_client.py <path_to_image>")
        sys.exit(1)

    result = query_vlm_plate(sys.argv[1])
    print("RAW   :", result)
    print("CLEAN :", clean_prediction(result))