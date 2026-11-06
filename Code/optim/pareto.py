"""Outils multi-objectifs : dominance, tri non dominé, encombrement, indicateurs.

Convention : tous les objectifs sont à MINIMISER.
"""
import numpy as np


def dominates(a, b):
    return bool(np.all(a <= b) and np.any(a < b))


def non_dominated_sort(F):
    """Tri non dominé rapide (Deb et al., 2002). Renvoie la liste des fronts (listes d'indices)."""
    n = len(F)
    S = [[] for _ in range(n)]
    cnt = np.zeros(n, dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            if dominates(F[i], F[j]):
                S[i].append(j)
                cnt[j] += 1
            elif dominates(F[j], F[i]):
                S[j].append(i)
                cnt[i] += 1
    fronts = [[i for i in range(n) if cnt[i] == 0]]
    k = 0
    while fronts[k]:
        nxt = []
        for i in fronts[k]:
            for j in S[i]:
                cnt[j] -= 1
                if cnt[j] == 0:
                    nxt.append(j)
        k += 1
        fronts.append(nxt)
    return fronts[:-1]


def crowding_distance(F):
    """Distance d'encombrement d'un ensemble de points d'un même front."""
    n, m = F.shape
    d = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for k in range(m):
        o = np.argsort(F[:, k])
        span = F[o[-1], k] - F[o[0], k]
        d[o[0]] = d[o[-1]] = np.inf
        if span > 0:
            d[o[1:-1]] += (F[o[2:], k] - F[o[:-2], k]) / span
    return d


def pareto_front(F):
    """Indices des points non dominés de F."""
    return non_dominated_sort(np.asarray(F))[0] if len(F) else []


def hypervolume_2d(F, ref):
    """Hypervolume (aire dominée) d'un ensemble 2D par rapport au point de référence ref."""
    F = np.asarray(F, dtype=float)
    F = F[np.all(F < ref, axis=1)]
    if len(F) == 0:
        return 0.0
    F = F[pareto_front(F)]
    F = F[np.argsort(F[:, 0])]
    hv, prev = 0.0, ref[1]
    for f1, f2 in F:
        hv += (ref[0] - f1) * (prev - f2)
        prev = f2
    return hv


def igd(F_found, F_true):
    """Distance générationnelle inversée : distance moyenne de chaque point du vrai front au plus proche trouvé."""
    if len(F_found) == 0:
        return np.inf
    D = np.linalg.norm(F_true[:, None, :] - np.asarray(F_found)[None, :, :], axis=2)
    return float(D.min(axis=1).mean())


class Normalizer:
    """Ramène les objectifs à [0, 1] sur l'ensemble faisable (pour l'hypervolume et la somme pondérée)."""

    def __init__(self, F):
        self.lo = F.min(axis=0)
        self.hi = F.max(axis=0)

    def __call__(self, F):
        return (np.asarray(F) - self.lo) / (self.hi - self.lo)
