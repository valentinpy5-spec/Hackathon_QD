"""Itération V3, lot L6 : un problème mis à l'échelle, plus proche d'un cas réel.

Le profil n'est plus un NACA à 4 paramètres mais une forme CST (Kulfan, 2008) à 10 coefficients :
    y_ext(x) = C(x) S_ext(x),  y_int(x) = C(x) S_int(x),  C(x) = sqrt(x) (1 - x),
    S(x) = sum_{i=0}^{4} A_i K_i x^i (1 - x)^(4 - i)  (polynômes de Bernstein, K_i = binomial(4, i)).
Chaque coefficient prend 8 niveaux (3 bits) entre des bornes qui couvrent des profils NACA de 0 à 6 %
de cambrure et 8 à 18 % d'épaisseur : 30 bits, 8^10 ~ 10^9 profils. Ni la grille ni le QUBO ne
s'énumèrent : il faut des solveurs heuristiques, comme sur un vrai problème.

Les solveurs sont des boîtes noires : XFOIL (croisière C_L = 0,4 et rampe de manœuvre, comme en V2) et
la SER par optique physique (secteur S2, 81 directions). La méthode des moments est trop lente pour
des milliers d'évaluations : c'est une limite assumée (la PO garde le classement, corrélation 0,999).
Contraintes : épaisseur maximale >= 8 % (calculée sur la géométrie), C_L,max >= 1,2, convergence
d'XFOIL, géométrie valide (intrados sous l'extrados).

Chaque évaluation est mise en cache (resultats/cache_cst.jsonl) : un même profil n'est simulé qu'une
fois, toutes méthodes et graines confondues ; le compteur de simulations de chaque exécution, lui,
compte les profils distincts demandés, comme CountedProblem.
"""
import fcntl
import json
import os
from math import comb

import numpy as np

from aero.naca4 import naca4
from aero.xfoil_runner import run_xfoil
from ser.po2d import echo_width, radar_directions, to_db, wavenumber

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(HERE, "resultats", "cache_cst.jsonl")
N_BERN = 4
N_VAR = 2 * (N_BERN + 1)
LEVELS = 8
RE, CL_CRUISE, CL_MAX_MIN, T_MIN = 3e6, 0.4, 1.2, 0.08
MAN_ALPHAS = (4, 6, 8, 9, 10, 11, 12, 13, 14)
SECTEUR = (-60.0, 0.0)                      # S2 : radar en contrebas, comme le scénario étudié en V2
FREQ = 10e9


# ------------------------------------------------------------ géométrie CST
def bernstein_matrix(x, N=N_BERN):
    return np.column_stack([comb(N, i) * x ** i * (1 - x) ** (N - i) for i in range(N + 1)])


def classe(x):
    return np.sqrt(x) * (1 - x)


def cst_coords(Au, Al, n_half=120):
    """Contour fermé, même convention que naca4 : BF extrados -> BA -> BF intrados."""
    beta = np.linspace(0, np.pi, n_half + 1)
    x = 0.5 * (1 - np.cos(beta))
    yu = classe(x) * (bernstein_matrix(x) @ Au)
    yl = classe(x) * (bernstein_matrix(x) @ Al)
    X = np.concatenate([x[::-1], x[1:]])
    Y = np.concatenate([yu[::-1], yl[1:]])
    return X, Y


def fit_cst(X, Y):
    """Coefficients CST qui approchent un contour donné (moindres carrés, extrados et intrados)."""
    n = (len(X) - 1) // 2
    xu, yu = X[:n + 1][::-1], Y[:n + 1][::-1]
    xl, yl = X[n:], Y[n:]
    out = []
    for x, y in ((xu, yu), (xl, yl)):
        msk = (x > 1e-4) & (x < 1 - 1e-4)
        M = classe(x[msk])[:, None] * bernstein_matrix(x[msk])
        out.append(np.linalg.lstsq(M, y[msk], rcond=None)[0])
    return out


def bornes():
    """Bornes de chaque coefficient : enveloppe des CST de NACA (m 0-6 %, p 0,3-0,5, t 8-18 %), +10 %."""
    A = []
    for m in (0.0, 0.02, 0.04, 0.06):
        for p in (0.3, 0.4, 0.5):
            for t in (0.08, 0.10, 0.12, 0.15, 0.18):
                Au, Al = fit_cst(*naca4(m, p, t, n_half=200))
                A.append(np.concatenate([Au, Al]))
    A = np.array(A)
    lo, hi = A.min(0), A.max(0)
    marge = 0.1 * (hi - lo)
    return lo - marge, hi + marge


LO, HI = bornes()
VALEURS = [np.linspace(LO[i], HI[i], LEVELS) for i in range(N_VAR)]
KEYS = tuple(f"a{i}" for i in range(N_VAR))
SPEC = dict(bits=(3,) * N_VAR, levels=(LEVELS,) * N_VAR, keys=KEYS, t_var=None,
            mlp_epochs=800, mlp_models=3)


def coefficients(idx):
    A = np.array([VALEURS[i][j] for i, j in enumerate(idx)])
    return A[:N_BERN + 1], A[N_BERN + 1:]


def init_aleatoire(rng, n):
    return [tuple(int(v) for v in rng.integers(0, LEVELS, N_VAR)) for _ in range(n)]


SPEC["init"] = init_aleatoire


# ------------------------------------------------------------ évaluation (boîtes noires)
def evaluer(idx):
    Au, Al = coefficients(idx)
    X, Y = cst_coords(Au, Al, n_half=120)
    n = 120
    epaisseur = Y[:n + 1][::-1] - Y[n:]                  # y_ext(x) - y_int(x) sur la même grille
    geom_ok = bool(np.all(epaisseur[1:-1] > 0))
    tmax = float(epaisseur.max())
    row = dict(zip(KEYS, (int(v) for v in idx)), t_max=tmax, geom_ok=geom_ok, alpha=np.nan,
               CD_counts=np.nan, CLmax=np.nan, valid=False, SER_db=np.nan)
    if geom_ok:
        cr = run_xfoil(0, 0, 0, [], RE, coords=(X, Y), cls=[CL_CRUISE], timeout=20)
        ok = len(cr["alpha"]) == 1
        ma = run_xfoil(0, 0, 0, list(MAN_ALPHAS), RE, coords=(X, Y), timeout=30)
        row["CLmax"] = float(np.max(ma["CL"])) if len(ma["CL"]) else np.nan
        row["valid"] = ok
        if ok:
            row["alpha"] = float(cr["alpha"][0])
            row["CD_counts"] = float(cr["CD"][0]) * 1e4
        Xs, Ys = cst_coords(Au, Al, n_half=200)
        lo, hi = SECTEUR
        eps = np.linspace(-(hi - lo) / 2, (hi - lo) / 2, 81)
        a = (row["alpha"] if ok else 0.0) - 0.5 * (hi + lo)
        row["SER_db"] = float(to_db(echo_width(Xs, Ys, radar_directions(a, eps), wavenumber(FREQ)).mean()))
    cl = row["CLmax"] if np.isfinite(row["CLmax"]) else 0.0
    g = [T_MIN - tmax, CL_MAX_MIN - cl, 0.0 if row["valid"] else 1.0, 0.0 if geom_ok else 1.0]
    row["cv"] = float(sum(max(0.0, gi) for gi in g))
    row["feasible"] = bool(row["cv"] <= 0)
    return row


def _charge_cache():
    cache = {}
    if os.path.exists(CACHE):
        with open(CACHE) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cache[tuple(r[k] for k in KEYS)] = r
    return cache


class ProblemeSimule:
    """Même interface que optim.problem.CountedProblem, mais les profils sont simulés à la demande."""

    def __init__(self):
        self.cache = _charge_cache()
        self.table = {}
        self.seen, self._set = [], set()

    def __call__(self, idx):
        idx = tuple(int(i) for i in idx)
        if idx not in self.table:
            r = self.cache.get(idx)
            if r is None:
                r = evaluer(idx)
                with open(CACHE, "a") as f:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    f.write(json.dumps(r) + "\n")
                    fcntl.flock(f, fcntl.LOCK_UN)
                self.cache[idx] = r
            self.table[idx] = r
        if idx not in self._set:
            self._set.add(idx)
            self.seen.append(idx)
        return self.table[idx]

    @property
    def n_sims(self):
        return len(self.seen)

    def F(self, idx):
        r = self(idx)
        return np.array([r["CD_counts"], r["SER_db"]]), r["cv"]
