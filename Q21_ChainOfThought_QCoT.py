#!/usr/bin/env python3
"""
Q21 Chain of Thought — tehnika: Quantum Chain-of-Thought (QCoT)
(čisto kvantno: sekvencijalna kompozicija unitarnih „reasoning-lens-ova").

Koncept (kvantni analog klasičnog CoT-a „let's think step by step"):
  - Početna misao: |ψ_0⟩ = amplitude-encoding globalnog freq_vector-a CELOG CSV-a.
  - Za s = 1..S primenjuje se „thought" unitar U_s (Ry-sloj + ring-CNOT),
    čiji su parametri deterministički izvedeni iz jedne od 5 SEMANTIČKI
    RAZLIČITIH feature-funkcija nad CELIM CSV-om:
        F1 HOT     : globalni freq_vector.
        F2 RECENT  : freq_vector poslednjih K redova.
        F3 PAIR    : za svaki broj i, maksimalna pair-co-occurrence sa j≠i.
        F4 PARITY  : koliko puta se i pojavio u kombinaciji sa parnom ukupnom sumom.
        F5 GAP     : prosečan min-gap do susednog broja u istoj kombinaciji.
  - Lanac: |ψ_S⟩ = U_S · U_{S−1} · … · U_1 · |ψ_0⟩ (bez aux registra, bez interferencije
    kroz template-e — čisti sekvencijalni reasoning-lanac, za razliku od Q20).
  - Readout: p = |ψ_S|² → bias_39 → TOP-7 = NEXT.

Razlika u odnosu na slične fajlove:
  Q11 (Transformer): ponavlja ISTI attention+FFN mehanizam kroz slojeve.
  Q12 (QARLM):       token-autoregresija (po jedan broj), resetuje kontekst.
  Q20 (QPTM):        PARALELNA superpozicija 4 template-a preko aux-a.
  QCoT:              SEKVENCIJALNA kompozicija 5 semantički različitih lens-unitara
                     (pravi „chain" — svaka misao transformiše prethodnu).

Sve deterministički: seed=39; ceo CSV koristi se u svim feature-funkcijama.
Deterministička grid-optimizacija (nq, S, K) po cos(bias_39, freq_csv).

Okruženje: Python 3.11.13, qiskit 1.4.4, qiskit-machine-learning 0.8.3, macOS M1 (vidi README.md).
"""

from __future__ import annotations

import csv
import random
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    from scipy.sparse import SparseEfficiencyWarning

    warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
except ImportError:
    pass

from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector

# =========================
# Seed
# =========================
SEED = 39
np.random.seed(SEED)
random.seed(SEED)
try:
    from qiskit_machine_learning.utils import algorithm_globals

    algorithm_globals.random_seed = SEED
except ImportError:
    pass

# =========================
# Konfiguracija
# =========================
CSV_PATH = Path("/Users/4c/Desktop/GHQ/data/loto7hh_4600_k31.csv")
N_NUMBERS = 7
N_MAX = 39

GRID_NQ = (5, 6)
GRID_S = (1, 2, 3, 4, 5)
GRID_K = (100, 500, 2000)


# =========================
# CSV
# =========================
def load_rows(path: Path) -> np.ndarray:
    rows: List[List[int]] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        if not header or "Num1" not in header[0]:
            f.seek(0)
            r = csv.reader(f)
            next(r, None)
        for row in r:
            if not row or row[0].strip() == "Num1":
                continue
            rows.append([int(row[i]) for i in range(N_NUMBERS)])
    return np.array(rows, dtype=int)


def freq_vector(H: np.ndarray) -> np.ndarray:
    c = np.zeros(N_MAX, dtype=np.float64)
    for v in H.ravel():
        if 1 <= v <= N_MAX:
            c[int(v) - 1] += 1.0
    return c


def amp_from_freq(f: np.ndarray, nq: int) -> np.ndarray:
    dim = 2 ** nq
    edges = np.linspace(0, N_MAX, dim + 1, dtype=int)
    amp = np.array(
        [float(f[edges[i] : edges[i + 1]].mean()) if edges[i + 1] > edges[i] else 0.0 for i in range(dim)],
        dtype=np.float64,
    )
    amp = np.maximum(amp, 0.0)
    n2 = float(np.linalg.norm(amp))
    if n2 < 1e-18:
        amp = np.ones(dim, dtype=np.float64) / np.sqrt(dim)
    else:
        amp = amp / n2
    return amp


# =========================
# 5 reasoning feature-a (svaki vraća vektor dužine N_MAX)
# =========================
def feature_hot(H: np.ndarray) -> np.ndarray:
    return freq_vector(H)


def feature_recent(H: np.ndarray, K: int) -> np.ndarray:
    K_eff = max(N_NUMBERS, min(H.shape[0], int(K)))
    return freq_vector(H[-K_eff:])


def feature_pair(H: np.ndarray) -> np.ndarray:
    """Za svako i: max_{j≠i} P[i,j] (najjači partner broja i)."""
    P = np.zeros((N_MAX, N_MAX), dtype=np.float64)
    for row in H:
        for a in row:
            for b in row:
                if a != b and 1 <= a <= N_MAX and 1 <= b <= N_MAX:
                    P[a - 1, b - 1] += 1.0
    f = np.zeros(N_MAX, dtype=np.float64)
    for i in range(N_MAX):
        row = P[i].copy()
        row[i] = 0.0
        f[i] = float(row.max()) if row.size else 0.0
    return f


def feature_parity(H: np.ndarray) -> np.ndarray:
    """Koliko puta se i pojavio u redu sa parnom ukupnom sumom."""
    f = np.zeros(N_MAX, dtype=np.float64)
    for row in H:
        s = int(np.sum(row))
        if s % 2 == 0:
            for v in row:
                if 1 <= v <= N_MAX:
                    f[int(v) - 1] += 1.0
    return f


def feature_gap(H: np.ndarray) -> np.ndarray:
    """Prosečan min-gap broja i do susednog elementa u istoj (sortiranoj) kombinaciji."""
    total = np.zeros(N_MAX, dtype=np.float64)
    cnt = np.zeros(N_MAX, dtype=np.float64)
    for row in H:
        s = np.sort(np.asarray(row, dtype=int))
        for idx, v in enumerate(s):
            if not (1 <= v <= N_MAX):
                continue
            gaps = []
            if idx > 0:
                gaps.append(abs(int(v) - int(s[idx - 1])))
            if idx < len(s) - 1:
                gaps.append(abs(int(s[idx + 1]) - int(v)))
            if gaps:
                total[int(v) - 1] += float(min(gaps))
                cnt[int(v) - 1] += 1.0
    out = np.zeros(N_MAX, dtype=np.float64)
    mask = cnt > 0
    out[mask] = total[mask] / cnt[mask]
    return out


FEATURES = (feature_hot, feature_recent, feature_pair, feature_parity, feature_gap)
FEATURE_NAMES = ("F1 HOT   ", "F2 RECENT", "F3 PAIR  ", "F4 PARITY", "F5 GAP   ")


def feature_vector(s_idx: int, H: np.ndarray, K: int) -> np.ndarray:
    fn = FEATURES[s_idx]
    if fn is feature_recent:
        return fn(H, K)
    return fn(H)


# =========================
# Deterministički Ry-uglovi po-qubit-u iz feature-a (dužina N_MAX)
# =========================
def feature_to_angles(f: np.ndarray, nq: int) -> np.ndarray:
    m = float(f.max())
    if m < 1e-18:
        return np.zeros(nq, dtype=np.float64)
    edges = np.linspace(0, N_MAX, nq + 1, dtype=int)
    angles = np.zeros(nq, dtype=np.float64)
    for k in range(nq):
        lo, hi = int(edges[k]), int(edges[k + 1])
        seg = f[lo:hi] if hi > lo else np.array([0.0])
        angles[k] = float(np.pi * (float(seg.mean()) / m))
    return angles


# =========================
# QCoT kolo: |ψ_0⟩ = amp(freq_csv) ; U_s za s = 1..S
# =========================
def build_qcot_state(H: np.ndarray, nq: int, S: int, K: int) -> Statevector:
    amp0 = amp_from_freq(freq_vector(H), nq)
    qc = QuantumCircuit(nq, name="qcot")
    qc.append(StatePreparation(amp0.tolist()), range(nq))

    for s_idx in range(int(S)):
        f_s = feature_vector(s_idx, H, K)
        theta = feature_to_angles(f_s, nq)
        for k in range(nq):
            qc.ry(float(theta[k]), k)
        for k in range(nq):
            qc.cx(k, (k + 1) % nq)

    return Statevector(qc)


def qcot_state_probs(H: np.ndarray, nq: int, S: int, K: int) -> np.ndarray:
    sv = build_qcot_state(H, nq, S, K)
    p = np.abs(sv.data) ** 2
    s = float(p.sum())
    return p / s if s > 0 else p


# =========================
# Readout
# =========================
def bias_39(probs: np.ndarray, n_max: int = N_MAX) -> np.ndarray:
    b = np.zeros(n_max, dtype=np.float64)
    for idx, p in enumerate(probs):
        b[idx % n_max] += float(p)
    s = float(b.sum())
    return b / s if s > 0 else b


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-18 or nb < 1e-18:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def pick_next_combination(probs: np.ndarray, k: int = N_NUMBERS, n_max: int = N_MAX) -> Tuple[int, ...]:
    b = bias_39(probs, n_max)
    order = np.argsort(-b, kind="stable")
    return tuple(sorted(int(o + 1) for o in order[:k]))


# =========================
# Determ. grid-optimizacija (nq, S, K)
# =========================
def optimize_hparams(H: np.ndarray):
    f_csv = freq_vector(H)
    s_tot = float(f_csv.sum())
    f_csv_n = f_csv / s_tot if s_tot > 0 else np.ones(N_MAX) / N_MAX
    best = None
    for nq in GRID_NQ:
        for S in GRID_S:
            for K in GRID_K:
                try:
                    p = qcot_state_probs(H, nq, int(S), int(K))
                    bi = bias_39(p)
                    score = cosine(bi, f_csv_n)
                except Exception:
                    continue
                key = (score, nq, int(S), -int(K))
                if best is None or key > best[0]:
                    best = (key, dict(nq=nq, S=int(S), K=int(K), score=float(score)))
    return best[1] if best else None


def main() -> int:
    H = load_rows(CSV_PATH)
    if H.shape[0] < 1:
        print("premalo redova")
        return 1

    print("Q21 Chain-of-Thought (QCoT — sekvencijalni reasoning-lanac): CSV:", CSV_PATH)
    print("redova:", H.shape[0], "| seed:", SEED, "| dostupnih lens-ova:", len(FEATURES))

    best = optimize_hparams(H)
    if best is None:
        print("grid optimizacija nije uspela")
        return 2
    print(
        "BEST hparam:",
        "nq=", best["nq"],
        "| S (koraka misli):", best["S"],
        "| K (recent):", best["K"],
        "| cos(bias, freq_csv):", round(float(best["score"]), 6),
    )

    nq_best = int(best["nq"])
    K_best = int(best["K"])

    f_csv = freq_vector(H)
    s_tot = float(f_csv.sum())
    f_csv_n = f_csv / s_tot if s_tot > 0 else np.ones(N_MAX) / N_MAX

    print("--- kumulativni lanac (nakon svakog koraka) ---")
    for S_cur in range(1, int(best["S"]) + 1):
        p_s = qcot_state_probs(H, nq_best, S_cur, K_best)
        cos_s = cosine(bias_39(p_s), f_csv_n)
        pred_s = pick_next_combination(p_s)
        print(f"S={S_cur:d}  +{FEATURE_NAMES[S_cur-1]}  cos={cos_s:.6f}  NEXT={pred_s}")

    p = qcot_state_probs(H, nq_best, int(best["S"]), K_best)
    pred = pick_next_combination(p)
    print("--- glavna predikcija (QCoT finalni lanac) ---")
    print("predikcija NEXT:", pred)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



"""
Q21 Chain-of-Thought (QCoT — sekvencijalni reasoning-lanac): CSV: /data/loto7hh_4600_k31.csv
redova: 4600 | seed: 39 | dostupnih lens-ova: 5
BEST hparam: nq= 6 | S (koraka misli): 1 | K (recent): 100 | cos(bias, freq_csv): 0.808989
--- kumulativni lanac (nakon svakog koraka) ---
S=1  +F1 HOT     cos=0.808989  NEXT=(2, 9, 10, 12, 19, 21, 24)
--- glavna predikcija (QCoT finalni lanac) ---
predikcija NEXT: (2, 9, 10, 12, 19, 21, 24)
"""



"""
Q21_ChainOfThought_QCoT.py — tehnika: Quantum Chain-of-Thought (QCoT).

Koncept:
Lanac misli kao sekvencijalni unitarni proces. Početna misao je amplitude-encoding
globalnog freq_vector-a CELOG CSV-a. Zatim se primenjuje S različitih „thought"
unitara U_s (Ry-sloj + ring-CNOT), gde su parametri svakog U_s deterministički
izvedeni iz jedne od 5 semantički različitih feature-funkcija nad CELIM CSV-om:
  F1 HOT    — globalni freq_vector
  F2 RECENT — freq_vector poslednjih K redova
  F3 PAIR   — za svako i, max co-occurrence sa partnerom j≠i
  F4 PARITY — koliko puta je i u kombinaciji sa parnom ukupnom sumom
  F5 GAP    — prosečan min-gap broja i do susednog elementa u istoj kombinaciji

Kolo (nq qubit-a, bez aux-a):
  StatePreparation(|ψ_0⟩ = amp(freq_csv)).
  Za s = 1..S: Ry(theta_s[k]) + ring-CNOT, gde je theta_s deterministički dobijen
  iz feature-vektora F_s (po-qubit-u prosek segmenta · π / max(F_s)).
  Finalno stanje: |ψ_S⟩ = U_S·…·U_1·|ψ_0⟩.
Readout: p = |ψ_S|² → bias_39 → TOP-7 = NEXT.

Tehnike:
StatePreparation za početnu misao iz CELOG CSV-a.
Deterministički izveden Ry-sloj + ring-CNOT po koraku (PQC, ali sa fiksnim parametrima).
Sekvencijalna kompozicija unitara (čisti „chain", bez aux-a, bez interferencije template-a).
Egzaktni Statevector (bez uzorkovanja).
Deterministička grid-optimizacija (nq, S, K).

Prednosti:
Direktan kvantni analog CoT-a: „misli korak po korak" kroz različite lens-ove.
Svaki korak unosi SEMANTIČKI NOV feature (pair, parity, gap, …), a ne isti mehanizam.
Ceo CSV učestvuje u svakom feature-u (pravilo 10).
Čisto kvantno: bez klasičnog treninga, bez softmax-a, bez hibrida.

Nedostaci:
Sekvencijalni unitari bez aux-a → nema interferencije ili post-selekcije.
Ry-feature-to-angles mapiranje je deterministička heuristika (mean per-qubit-segment),
drugi izbor mapiranja bi dao drugi bias.
Fiksan redosled lens-ova (HOT → RECENT → PAIR → PARITY → GAP); permutacija nije optimizovana.
mod-39 readout meša stanja (dim 2^nq ≠ 39).
"""
