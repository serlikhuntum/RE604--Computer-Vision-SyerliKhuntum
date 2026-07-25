"""
label_parser.py
----------------
Dataset ini menggunakan format label YOLO per-karakter, contoh isi file .txt:

    11 0.088040 0.328042 0.116279 0.592593
    9  0.274086 0.362434 0.109635 0.640212
    ...

Setiap baris merepresentasikan SATU karakter pada plat nomor:
    <class_id> <x_center> <y_center> <width> <height>   (semua ternormalisasi 0-1)

Class id -> karakter:
    0-9   -> digit '0'-'9'
    10-35 -> huruf 'A'-'Z'  (10='A', 11='B', ..., 35='Z')

Ground truth plat direkonstruksi dengan cara:
    1. Urutkan seluruh karakter berdasarkan koordinat x_center (kiri -> kanan).
    2. Gabungkan karakter yang berurutan dan bertipe sama (huruf/angka) menjadi
       satu blok, lalu pisahkan antar blok dengan spasi.
       Contoh: B, 9,0,6,2, V,E,H  ->  "B 9062 VEH"

Ini sesuai format umum plat nomor Indonesia: [huruf wilayah] [nomor] [huruf seri].
"""

from pathlib import Path

CLASS_TO_CHAR = {i: str(i) for i in range(10)}
CLASS_TO_CHAR.update({10 + i: chr(ord('A') + i) for i in range(26)})


def parse_yolo_label(txt_path: str) -> str:
    """Baca file label YOLO per-karakter dan kembalikan string plat nomor ground truth."""
    chars = []  # list of (x_center, char)

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            class_id = int(parts[0])
            x_center = float(parts[1])
            char = CLASS_TO_CHAR.get(class_id)
            if char is None:
                continue  # abaikan class id yang tidak dikenal
            chars.append((x_center, char))

    # urutkan kiri -> kanan
    chars.sort(key=lambda t: t[0])

    if not chars:
        return ""

    # kelompokkan karakter berurutan yang bertipe sama (huruf vs angka) menjadi blok
    blocks = []
    current_block = chars[0][1]
    current_is_digit = chars[0][1].isdigit()

    for _, ch in chars[1:]:
        is_digit = ch.isdigit()
        if is_digit == current_is_digit:
            current_block += ch
        else:
            blocks.append(current_block)
            current_block = ch
            current_is_digit = is_digit
    blocks.append(current_block)

    return " ".join(blocks)


def label_path_for_image(image_path: str) -> str:
    """Kembalikan path file .txt yang berpasangan dengan file gambar."""
    return str(Path(image_path).with_suffix(".txt"))


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python label_parser.py <path_to_label.txt>")
        sys.exit(1)

    print(parse_yolo_label(sys.argv[1]))
