"""
cer.py
------
Implementasi Character Error Rate (CER) berbasis Levenshtein Distance.

    CER = (S + D + I) / N

    S = jumlah substitusi karakter
    D = jumlah penghapusan (deletion) karakter
    I = jumlah penyisipan (insertion) karakter
    N = jumlah karakter pada ground truth

Perhitungan S, D, I didapat dari traceback matriks Dynamic Programming
Levenshtein Distance antara string prediksi dan ground truth.
"""

from typing import Tuple


def _levenshtein_ops(ref: str, hyp: str) -> Tuple[int, int, int]:
    """Hitung jumlah substitusi (S), deletion (D), dan insertion (I)
    untuk mengubah `hyp` (prediksi) menjadi `ref` (ground truth).
    """
    n, m = len(ref), len(hyp)

    # dp[i][j] = edit distance minimal antara ref[:i] dan hyp[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i  # butuh i deletion untuk kosongkan ref[:i]
    for j in range(m + 1):
        dp[0][j] = j  # butuh j insertion untuk bentuk hyp[:j] dari ""

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                substitution = dp[i - 1][j - 1] + 1
                deletion = dp[i - 1][j] + 1       # karakter ref[i-1] dihapus
                insertion = dp[i][j - 1] + 1       # karakter hyp[j-1] disisipkan
                dp[i][j] = min(substitution, deletion, insertion)

    # traceback untuk memisahkan jumlah S, D, I
    i, j = n, m
    S = D = I = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            i -= 1
            j -= 1
            continue
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            S += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            D += 1
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            I += 1
            j -= 1
        else:
            # fallback pengaman, seharusnya tidak tercapai
            break

    return S, D, I


def compute_cer(ground_truth: str, prediction: str, normalize: bool = True) -> dict:
    """Hitung CER antara ground_truth dan prediction.

    normalize=True akan menghapus spasi dan menyeragamkan huruf besar
    sebelum dibandingkan (umum dilakukan pada evaluasi OCR plat nomor,
    karena spasi bukan bagian dari "karakter" plat itu sendiri).

    Return dict berisi: S, D, I, N, CER
    """
    ref = ground_truth
    hyp = prediction

    if normalize:
        ref = ref.replace(" ", "").upper()
        hyp = hyp.replace(" ", "").upper()

    N = len(ref)
    if N == 0:
        # tidak ada ground truth valid untuk dibandingkan
        return {"S": 0, "D": 0, "I": len(hyp), "N": 0, "CER": None}

    S, D, I = _levenshtein_ops(ref, hyp)
    cer_score = (S + D + I) / N

    return {"S": S, "D": D, "I": I, "N": N, "CER": cer_score}


if __name__ == "__main__":
    example = compute_cer("B 9062 VEH", "B 9O62 VEH")
    print(example)
