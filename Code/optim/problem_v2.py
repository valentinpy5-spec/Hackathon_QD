"""Itération V2 : le problème reformulé.

Changements par rapport à V1 (voir la fiche projet) :
    - géométrie : NACA 4 chiffres MODIFIÉE, avec un indice de rayon de bord
      d'attaque I indépendant de l'épaisseur (aero.naca4.naca4_modified) ;
    - simulateur aérodynamique de référence : XFOIL (au lieu du modèle rapide) ;
    - vol de croisière à portance imposée (C_L = 0,4) : C_D et l'incidence
      de croisière, qui est aussi l'attitude vue par le radar ;
    - contrainte de manœuvre : portance maximale C_L,max >= 1,2, estimée sur
      une rampe d'incidence XFOIL (4 à 14°). Un nez trop aigu décroche tôt :
      c'est ce qui crée le conflit avec la SER, qui veut un nez aigu ;
    - SER : même optique physique, sur la nouvelle géométrie.

Grille : m (8 niveaux) x p (4) x t (8) x I (4) = 1 024 profils, 3 + 2 + 3 + 2 = 10 bits.

    python3 -m optim.problem_v2      # construit resultats/base_V2.csv (environ 10 min sur 10 cœurs)
"""
import csv
import os
from dataclasses import dataclass, field
from itertools import product
from multiprocessing import Pool

import numpy as np

from aero.naca4 import naca4_modified, leading_edge_radius_modified
from aero.xfoil_runner import run_xfoil
from ser.model import RadarCase, evaluate_ser

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLE = os.path.join(HERE, "resultats", "base_V2.csv")
KEYS = ("im", "ip", "it", "iI")


@dataclass
class SpaceV2:
    m_levels: tuple = tuple(np.arange(8) * 0.01)               # 0 à 7 %
    p_levels: tuple = (0.25, 0.35, 0.45, 0.55)
    t_levels: tuple = tuple(0.06 + np.arange(8) * 0.02)        # 6 à 20 %
    I_levels: tuple = (2.0, 4.0, 6.0, 8.0)                     # rayon de BA : (I/6)^2 = 0,11 à 1,78 fois le NACA standard

    @property
    def levels(self):
        return (self.m_levels, self.p_levels, self.t_levels, self.I_levels)

    @property
    def n_levels(self):
        return tuple(len(v) for v in self.levels)

    @property
    def bits(self):
        return tuple(int(np.log2(n)) for n in self.n_levels)

    def values(self, idx):
        return tuple(float(lv[i]) for lv, i in zip(self.levels, idx))

    def all_indices(self):
        return list(product(*[range(n) for n in self.n_levels]))


@dataclass
class CaseV2:
    Re: float = 3e6
    CL_cruise: float = 0.4
    CL_max_min: float = 1.2                  # exigence de manœuvre
    manoeuvre_alphas: tuple = (4, 6, 8, 9, 10, 11, 12, 13, 14)
    t_min: float = 0.08
    T: float = 0.3                           # position de l'épaisseur maximale (fixée en V2)
    sectors: dict = field(default_factory=lambda: {"S1": (-20.0, 20.0), "S2": (-60.0, 0.0)})


def ser_sector(m, p, t, I, T, alpha, lo, hi):
    rc = RadarCase(elev_max=0.5 * (hi - lo))
    return evaluate_ser(m, p, t, alpha=alpha - 0.5 * (hi + lo), case=rc, I=I, T=T).ser_db


def simulate_v2(idx, space=None, case=None):
    space = space or SpaceV2()
    case = case or CaseV2()
    m, p, t, I = space.values(idx)
    X, Y = naca4_modified(m, p, t, I, case.T, n_half=120)
    cr = run_xfoil(m, p, t, [], case.Re, coords=(X, Y), cls=[case.CL_cruise], timeout=20)
    ok = len(cr["alpha"]) == 1
    ma = run_xfoil(m, p, t, list(case.manoeuvre_alphas), case.Re, coords=(X, Y), timeout=30)
    clmax = float(np.max(ma["CL"])) if len(ma["CL"]) else np.nan
    alpha = float(cr["alpha"][0]) if ok else np.nan
    row = dict(im=idx[0], ip=idx[1], it=idx[2], iI=idx[3], m=m, p=p, t=t, I=I,
               r_le=leading_edge_radius_modified(t, I), alpha=alpha,
               CL=case.CL_cruise, CD=float(cr["CD"][0]) if ok else np.nan,
               CD_counts=float(cr["CD"][0]) * 1e4 if ok else np.nan,
               xtr_up=float(cr["xtr_up"][0]) if ok else np.nan,
               CLmax=clmax, alpha_clmax=float(ma["alpha"][np.argmax(ma["CL"])]) if len(ma["CL"]) else np.nan,
               n_man=int(len(ma["alpha"])), valid=ok)
    for name, (lo, hi) in case.sectors.items():
        row[f"SER_{name}"] = ser_sector(m, p, t, I, case.T, alpha if ok else 0.0, lo, hi)
    g = [case.t_min - t, case.CL_max_min - (clmax if np.isfinite(clmax) else 0.0), 0.0 if ok else 1.0]
    row["cv"] = float(sum(max(0.0, gi) for gi in g))
    row["feasible"] = bool(row["cv"] <= 0)
    row["SER_db"] = row["SER_S1"]                          # objectif par défaut : scénario de la note
    return row


def _one(idx):
    return simulate_v2(idx)


def build_table_v2(path=TABLE, workers=10):
    if os.path.exists(path):
        rows = list(csv.DictReader(open(path)))
        out = []
        for r in rows:
            d = {}
            for k, v in r.items():
                if k in ("valid", "feasible"):
                    d[k] = v == "True"
                elif k in KEYS or k == "n_man":
                    d[k] = int(v)
                else:
                    d[k] = float(v)
            out.append(d)
        return out
    with Pool(workers) as pool:
        rows = pool.map(_one, SpaceV2().all_indices(), chunksize=2)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return build_table_v2(path)


MOM = os.path.join(HERE, "resultats", "ser_mom_V2.csv")


def with_objective(table, sector="S1", source="PO"):
    """Copie de la table où SER_db est la SER du secteur choisi.

    source = "PO" (optique physique, colonnes de base_V2.csv) ou "MoM" (méthode des moments
    TM, resultats/ser_mom_V2.csv produit par python3 -m ser.compare_po_mom grille).
    La MoM est la référence depuis la V2 : elle corrige l'optique physique sur les nez aigus."""
    mom = None
    if source == "MoM":
        mom = {tuple(int(r[k]) for k in KEYS): r for r in csv.DictReader(open(MOM))}
    out = []
    for r in table:
        d = dict(r)
        d["SER_db"] = r[f"SER_{sector}"] if mom is None else float(mom[tuple(r[k] for k in KEYS)][f"SER_{sector}_MoM"])
        out.append(d)
    return out


if __name__ == "__main__":
    import time
    t0 = time.time()
    T = build_table_v2()
    feas = [r for r in T if r["feasible"]]
    print(f"{len(T)} profils en {time.time() - t0:.0f} s ; croisière convergée {sum(r['valid'] for r in T)}, "
          f"faisables {len(feas)}")
