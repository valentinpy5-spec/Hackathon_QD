"""Espace de conception discret et simulateur « compté ».

Pourquoi discret : la formulation QUBO travaille sur des bits. On fixe donc
dès la phase 1 une grille de niveaux pour (m, p, t), et c'est CETTE grille que
partagent NSGA-II (variables entières), la boucle QUBO (bits) et le calcul du
front de Pareto exact (énumération exhaustive). Si les trois approches ne
travaillent pas sur la même grille, la comparaison n'a plus de sens.

Encodage binaire : chaque variable prend 2^k niveaux, codés sur k bits,
    x = x_min + dx * sum_j 2^j b_j .
Un polynôme quadratique en (m, p, t) reste quadratique en bits, car
x_i x_l = sum_j sum_k 2^(j+k) dx_i dx_l b_ij b_lk (+ termes linéaires) et
b^2 = b. C'est ce qui permet de passer du modèle de substitution au QUBO.
"""
from dataclasses import dataclass, field
from itertools import product

import numpy as np

from .model import evaluate, evaluate_at_cl


@dataclass
class DesignSpace:
    m_levels: np.ndarray = field(default_factory=lambda: np.arange(8) * 0.01)          # 0 à 7 %
    p_levels: np.ndarray = field(default_factory=lambda: 0.20 + np.arange(8) * 0.05)   # 0.20 à 0.55
    t_levels: np.ndarray = field(default_factory=lambda: 0.06 + np.arange(8) * 0.02)   # 6 à 20 %

    @property
    def levels(self):
        return (self.m_levels, self.p_levels, self.t_levels)

    @property
    def bits_per_var(self):
        nb = [int(np.log2(len(lv))) for lv in self.levels]
        assert all(2**b == len(lv) for b, lv in zip(nb, self.levels)), "2^k niveaux requis"
        return nb

    @property
    def n_bits(self):
        return sum(self.bits_per_var)

    @property
    def size(self):
        return int(np.prod([len(lv) for lv in self.levels]))

    def values(self, idx):
        """Indices entiers (im, ip, it) -> valeurs physiques (m, p, t)."""
        return tuple(float(lv[i]) for lv, i in zip(self.levels, idx))

    def encode(self, idx):
        """(im, ip, it) -> vecteur de bits, poids faible en premier."""
        bits = []
        for i, nb in zip(idx, self.bits_per_var):
            bits += [(i >> j) & 1 for j in range(nb)]
        return np.array(bits, dtype=int)

    def decode(self, bits):
        idx, k = [], 0
        for nb in self.bits_per_var:
            idx.append(int(sum(int(bits[k + j]) << j for j in range(nb))))
            k += nb
        return tuple(idx)

    def all_indices(self):
        return product(*[range(len(lv)) for lv in self.levels])


@dataclass
class FlightCase:
    """Point de fonctionnement commun à toutes les phases (à figer en équipe)."""
    Re: float = 3e6
    alpha: float = 2.0          # incidence imposée (mode 'alpha')
    CL_target: float = 0.4      # portance imposée (mode 'cl')
    mode: str = "alpha"         # 'alpha' ou 'cl'
    CL_min: float = 0.3         # contrainte de portance (mode 'alpha')
    t_min: float = 0.08         # contrainte d'épaisseur
    n_half: int = 100


class AeroSimulator:
    """Enveloppe du modèle aéro : cache + compteur de simulations.

    Le nombre d'appels UNIQUES (n_sims) est la métrique de comparaison entre
    NSGA-II et la boucle QUBO : il faut donc le compter au même endroit pour
    les deux.
    """

    def __init__(self, space=None, case=None):
        self.space = space or DesignSpace()
        self.case = case or FlightCase()
        self.cache = {}
        self.history = []   # ordre des simulations réellement lancées

    @property
    def n_sims(self):
        return len(self.cache)

    def __call__(self, idx):
        idx = tuple(int(i) for i in idx)
        if idx not in self.cache:
            m, p, t = self.space.values(idx)
            c = self.case
            if c.mode == "cl":
                r = evaluate_at_cl(m, p, t, c.CL_target, Re=c.Re, n_half=c.n_half)
            else:
                r = evaluate(m, p, t, alpha=c.alpha, Re=c.Re, n_half=c.n_half)
            self.cache[idx] = r
            self.history.append(idx)
        return self.cache[idx]

    def constraints(self, r):
        """g <= 0 (convention pymoo). Renvoie la liste des contraintes."""
        c = self.case
        g = [c.t_min - r.t]
        if c.mode == "alpha":
            g.append(c.CL_min - r.CL)
        g.append(0.0 if r.valid else 1.0)   # profil décollé ou calcul raté = infaisable
        return g
