# OCR Plat Nomor Kendaraan menggunakan Visual Language Model (VLM) via LM Studio

Program ini melakukan **Optical Character Recognition (OCR)** pada gambar plat nomor
kendaraan menggunakan **Visual Language Model (VLM)** `qwen2.5-vl-7b-instruct` yang
dijalankan secara lokal melalui **LM Studio**, lalu diintegrasikan dengan Python melalui
endpoint REST API (OpenAI-compatible) yang disediakan LM Studio.

Hasil prediksi dievaluasi menggunakan metrik **Character Error Rate (CER)**.

---

## 1. Struktur Proyek

```
anpr_vlm_ocr/
├── dataset/
│   └── test/                  # taruh dataset di sini (pasangan file .jpg + .txt)
├── results/
│   └── results.csv            # output evaluasi (dibuat otomatis)
├── src/
│   ├── label_parser.py        # rekonstruksi ground truth dari label YOLO per-karakter
│   ├── cer.py                 # implementasi metrik Character Error Rate (CER)
│   ├── lmstudio_client.py     # client untuk memanggil VLM via LM Studio REST API
│   ├── main.py                # program utama (inferensi + evaluasi + simpan CSV)
│   └── evaluate_summary.py    # ringkasan hasil (opsional, untuk analisis)
├── requirements.txt
└── README.md
```

## 2. Format Dataset

Dataset berisi pasangan file per sampel:

- `xxxx.jpg` -> gambar plat nomor
- `xxxx.txt` -> label **YOLO per-karakter**, satu baris per karakter:

  ```
  <class_id> <x_center> <y_center> <width> <height>
  ```

  Mapping `class_id` -> karakter:
  - `0–9`  -> digit `'0'`–`'9'`
  - `10–35` -> huruf `'A'`–`'Z'` (10='A', 11='B', ..., 35='Z')

  Ground truth string plat direkonstruksi otomatis oleh `src/label_parser.py` dengan
  mengurutkan karakter dari kiri ke kanan berdasarkan `x_center`, lalu mengelompokkan
  huruf dan angka yang berurutan menjadi blok (mis. `B 9062 VEH`).

Letakkan dataset di `dataset/test/` (atau folder lain, sesuaikan argumen `--dataset`).

## 3. Persiapan LM Studio

1. Buka aplikasi **LM Studio**.
2. Download & load model VLM **`qwen2.5-vl-7b-instruct`** (tab *Search* → cari model →
   *Download*, lalu *Load Model*).
3. Buka tab **Developer / Local Server**, nyalakan **Status: Running**.
4. Pastikan server reachable, contoh: `http://127.0.0.1:1234`
   (bisa dicek lewat tombol *Server Settings* / endpoint `GET /api/v1/models`).

LM Studio menyediakan endpoint yang kompatibel dengan OpenAI API di:
`http://127.0.0.1:1234/v1/chat/completions`, yang digunakan oleh `src/lmstudio_client.py`
untuk mengirim gambar (base64) beserta prompt teks ke model.

## 4. Instalasi Python

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 5. Menjalankan Program

```bash
python src/main.py \
    --dataset dataset/test \
    --output results/results.csv \
    --base-url http://127.0.0.1:1234 \
    --model qwen2.5-vl-7b-instruct
```

Argumen (semua opsional, punya default):

| Argumen      | Default                        | Keterangan                                   |
|--------------|---------------------------------|-----------------------------------------------|
| `--dataset`  | `dataset/test`                 | Folder berisi pasangan gambar + label         |
| `--output`   | `results/results.csv`          | Path file CSV hasil                           |
| `--base-url` | `http://127.0.0.1:1234`        | Alamat server LM Studio                       |
| `--model`    | `qwen2.5-vl-7b-instruct`       | Nama model VLM yang di-load di LM Studio      |
| `--prompt`   | *(prompt default OCR)*         | Prompt kustom yang dikirim ke model           |

Program akan:
1. Membaca setiap gambar di `dataset/test/`.
2. Merekonstruksi ground truth dari file label `.txt` pasangannya.
3. Mengirim gambar ke LM Studio dengan prompt:
   > "What is the license plate number shown in this image? Respond only with the plate number, no explanation, no extra text."
4. Menghitung CER antara prediksi dan ground truth.
5. Menyimpan seluruh hasil ke `results/results.csv` dengan kolom:
   `image, ground_truth, prediction, CER_score`
6. Menampilkan ringkasan (rata-rata CER, jumlah prediksi sempurna) di terminal.

## 6. Melihat Ringkasan Hasil (opsional)

```bash
python src/evaluate_summary.py --csv results/results.csv --top-n 3
```

Menampilkan rata-rata CER serta contoh kasus **sukses** (CER rendah) dan **gagal**
(CER tinggi) — berguna sebagai bahan penjelasan pada video proyek.

## 7. Metrik Evaluasi: Character Error Rate (CER)

```
CER = (S + D + I) / N
```

- **S** = jumlah karakter yang salah substitusi
- **D** = jumlah karakter ground truth yang terhapus / tidak muncul di prediksi
- **I** = jumlah karakter tambahan yang disisipkan pada prediksi
- **N** = jumlah karakter pada ground truth

Nilai `S`, `D`, `I` didapat dari traceback matriks **Levenshtein Distance** antara
string prediksi dan ground truth (lihat `src/cer.py`). Sebelum dibandingkan, spasi
dihapus dan huruf diseragamkan menjadi kapital, karena spasi hanyalah pemisah visual
antar blok plat, bukan bagian dari "karakter" yang sebenarnya.

Contoh:
- Ground truth: `B9062VEH`, Prediksi: `B9O62VEH` → 1 substitusi (`0`→`O`) → CER = 1/8 = 0.125
- CER = 0 berarti prediksi identik dengan ground truth (sempurna).

## 8. Catatan

- Model VLM harus benar-benar sudah **ter-load** di LM Studio sebelum menjalankan
  `main.py`, jika tidak permintaan API akan gagal.
- Kualitas prediksi sangat bergantung pada resolusi/kejelasan gambar plat dan
  kemampuan model VLM yang digunakan.
- Program ini murni memanggil API lokal (`127.0.0.1`) — tidak ada data yang
  dikirim ke internet / cloud.
