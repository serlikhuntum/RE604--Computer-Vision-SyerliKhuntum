"""
evaluate_summary.py
--------------------
Skrip bantu (opsional) untuk membaca file results.csv yang sudah dihasilkan
oleh main.py, lalu menampilkan:
    - rata-rata CER keseluruhan
    - jumlah prediksi sempurna (CER = 0)
    - contoh kasus SUKSES (CER rendah) dan GAGAL (CER tinggi)
      -> berguna untuk bahan penjelasan di video.

Cara pakai:
    python src/evaluate_summary.py --csv results/results.csv
"""

import argparse
import csv


def load_results(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["CER_score"] == "":
                continue
            row["CER_score"] = float(row["CER_score"])
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="results/results.csv")
    parser.add_argument("--top-n", type=int, default=3)
    args = parser.parse_args()

    rows = load_results(args.csv)
    if not rows:
        print("Tidak ada data valid pada CSV.")
        return

    mean_cer = sum(r["CER_score"] for r in rows) / len(rows)
    perfect = sum(1 for r in rows if r["CER_score"] == 0)

    rows_sorted = sorted(rows, key=lambda r: r["CER_score"])

    print("================ RINGKASAN HASIL ================")
    print(f"Jumlah sampel        : {len(rows)}")
    print(f"Rata-rata CER        : {mean_cer:.4f}")
    print(f"Prediksi sempurna    : {perfect} ({perfect/len(rows)*100:.2f}%)")

    print(f"\n--- Top {args.top_n} SUKSES (CER terendah) ---")
    for r in rows_sorted[: args.top_n]:
        print(f"  {r['image']:<20} GT='{r['ground_truth']}'  PRED='{r['prediction']}'  CER={r['CER_score']}")

    print(f"\n--- Top {args.top_n} GAGAL (CER tertinggi) ---")
    for r in rows_sorted[-args.top_n:][::-1]:
        print(f"  {r['image']:<20} GT='{r['ground_truth']}'  PRED='{r['prediction']}'  CER={r['CER_score']}")
    print("===================================================")


if __name__ == "__main__":
    main()
