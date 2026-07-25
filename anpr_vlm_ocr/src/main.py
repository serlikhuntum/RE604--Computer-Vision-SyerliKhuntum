import argparse
import csv
import sys
import time
from pathlib import Path

from label_parser import parse_yolo_label, label_path_for_image
from lmstudio_client import query_vlm_plate, clean_prediction, DEFAULT_PROMPT
from cer import compute_cer

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_image_files(dataset_dir: Path):
    files = [
        p for p in sorted(dataset_dir.iterdir())
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return files


def run(dataset_dir: str, output_csv: str, base_url: str, model: str, prompt: str):
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        print(f"[ERROR] Folder dataset tidak ditemukan: {dataset_dir}")
        sys.exit(1)

    image_files = find_image_files(dataset_path)
    if not image_files:
        print(f"[ERROR] Tidak ada file gambar (.jpg/.png) di {dataset_dir}")
        sys.exit(1)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_cer = 0.0
    valid_count = 0
    exact_match = 0

    print(f"Ditemukan {len(image_files)} gambar. Memulai inferensi ke LM Studio...")
    print(f"Endpoint : {base_url}/v1/chat/completions")
    print(f"Model    : {model}\n")

    # Buka file CSV di awal dan tulis setiap baris SEGERA setelah didapat.
    # Ini penting: kalau terjadi error di tengah/akhir proses (misalnya file
    # terkunci aplikasi lain, listrik mati, dsb), hasil yang sudah diproses
    # sejauh itu TETAP tersimpan, tidak hilang semua.
    try:
        csv_file = open(output_path, "w", newline="", encoding="utf-8")
    except PermissionError as e:
        print(f"[ERROR] Tidak bisa menulis ke {output_path}: {e}")
        print("Kemungkinan file itu sedang terbuka di Excel/aplikasi lain. Tutup dulu file itu, lalu coba lagi.")
        sys.exit(1)

    writer = csv.DictWriter(
        csv_file, fieldnames=["image", "ground_truth", "prediction", "CER_score"]
    )
    writer.writeheader()
    csv_file.flush()

    for idx, image_path in enumerate(image_files, start=1):
        label_path = Path(label_path_for_image(str(image_path)))
        ground_truth = parse_yolo_label(str(label_path)) if label_path.exists() else ""

        print(f"[{idx}/{len(image_files)}] {image_path.name} ... ", end="", flush=True)

        try:
            raw_pred = query_vlm_plate(
                str(image_path), base_url=base_url, model=model, prompt=prompt
            )
            prediction = clean_prediction(raw_pred)
        except Exception as e:
            print(f"GAGAL ({e})")
            prediction = ""
            raw_pred = f"[ERROR: {e}]"

        result = compute_cer(ground_truth, prediction)
        cer_score = result["CER"]

        if cer_score is not None:
            total_cer += cer_score
            valid_count += 1
            if cer_score == 0:
                exact_match += 1

        print(f"GT='{ground_truth}'  PRED='{prediction}'  CER={cer_score}")

        writer.writerow(
            {
                "image": image_path.name,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "CER_score": round(cer_score, 4) if cer_score is not None else "",
            }
        )
        csv_file.flush()  # tulis ke disk segera, jangan tunggu buffer penuh

    csv_file.close()

    # ringkasan
    mean_cer = total_cer / valid_count if valid_count else None
    accuracy = exact_match / valid_count if valid_count else None

    print("\n================ RINGKASAN EVALUASI ================")
    print(f"Total sampel dievaluasi : {valid_count}")
    print(f"Rata-rata CER           : {mean_cer:.4f}" if mean_cer is not None else "Rata-rata CER           : -")
    print(f"Exact match (CER=0)     : {exact_match} ({accuracy*100:.2f}%)" if accuracy is not None else "")
    print(f"Hasil disimpan di       : {output_path}")
    print("======================================================")


def parse_args():
    parser = argparse.ArgumentParser(
        description="OCR Plat Nomor Kendaraan menggunakan VLM via LM Studio"
    )
    parser.add_argument(
        "--dataset", type=str, default="dataset/test",
        help="Folder dataset berisi pasangan file .jpg + .txt (default: dataset/test)"
    )
    parser.add_argument(
        "--output", type=str, default="results/results.csv",
        help="Path file CSV output (default: results/results.csv)"
    )
    parser.add_argument(
        "--base-url", type=str, default="http://127.0.0.1:1234",
        help="Base URL server LM Studio (default: http://127.0.0.1:1234)"
    )
    parser.add_argument(
        "--model", type=str, default="smolvlm-500m-instruct",
        help="Nama model VLM yang sudah di-load di LM Studio"
    )
    parser.add_argument(
        "--prompt", type=str, default=DEFAULT_PROMPT,
        help="Prompt yang dikirim ke VLM"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    start = time.time()
    run(args.dataset, args.output, args.base_url, args.model, args.prompt)
    print(f"\nSelesai dalam {time.time() - start:.1f} detik.")