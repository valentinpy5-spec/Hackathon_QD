"""Le problème bi-objectif du hackathon : minimiser (C_D, SER) sous contraintes.

Un même objet sert aux trois méthodes (énumération exhaustive, NSGA-II, boucle
QUBO). Il compte les simulations DISTINCTES : c'est la métrique de comparaison.

Pour que les expériences répétées (dizaines de graines) restent rapides, on
évalue d'abord une fois les 512 profils (la base exhaustive, qui donne aussi le
front exact) puis chaque méthode « simule » en lisant cette table, à travers
le compteur. C'est strictement équivalent à appeler le simulateur, puisqu'il
est déterministe.
"""
import csv
import os
from dataclasses import dataclass, field, asdict

import numpy as np

from aero.design_space import DesignSpace, FlightCase
from aero.model import evaluate, evaluate_at_cl
from ser.model import RadarCase, evaluate_ser


@dataclass
class Scenario:
    name: str = "S1_note"
    flight: FlightCase = field(default_factory=FlightCase)      # alpha = 2°, CL >= 0.3, t >= 8 %
    radar: RadarCase = field(default_factory=RadarCase)          # 10 GHz, élévation [-20°, +20°]
    elev_min: float = None                                       # secteur asymétrique éventuel
    elev_max: float = None

    def radar_case(self):
        r = RadarCase(**{k: v for k, v in asdict(self.radar).items()})
        return r


def scenario(name):
    """Les deux scénarios étudiés dans les cours."""
    if name == "S1_note":
        return Scenario("S1_note")
    if name == "S2_sol":
        return Scenario("S2_sol", elev_min=-60.0, elev_max=0.0)
    raise ValueError(name)


def simulate(idx, space, sc):
    """Une « simulation » complète : aéro puis SER à l'attitude donnée par l'aéro."""
    m, p, t = space.values(idx)
    fc = sc.flight
    if fc.mode == "cl":
        a = evaluate_at_cl(m, p, t, fc.CL_target, Re=fc.Re, n_half=fc.n_half)
    else:
        a = evaluate(m, p, t, alpha=fc.alpha, Re=fc.Re, n_half=fc.n_half)
    rc = sc.radar_case()
    if sc.elev_min is not None:
        # secteur [elev_min, elev_max] : on recentre la grille d'élévations
        rc.elev_max = 0.5 * (sc.elev_max - sc.elev_min)
        offset = 0.5 * (sc.elev_max + sc.elev_min)
    else:
        offset = 0.0
    s = evaluate_ser(m, p, t, alpha=a.alpha - offset, case=rc)
    # alpha - offset : décaler toutes les élévations de +offset revient à décaler alpha de -offset
    g = [fc.t_min - t]
    if fc.mode == "alpha":
        g.append(fc.CL_min - a.CL)
    g.append(0.0 if a.valid else 1.0)
    return dict(im=idx[0], ip=idx[1], it=idx[2], m=m, p=p, t=t, alpha=a.alpha, CL=a.CL,
                CD=a.CD, CD_counts=a.CD * 1e4, SER_db=s.ser_db, SER_nose_db=s.ser_nose_db,
                SER_max_db=s.ser_max_db, valid=a.valid, cv=float(sum(max(0.0, gi) for gi in g)),
                feasible=all(gi <= 0 for gi in g), bits="".join(map(str, space.encode(idx))))


def build_table(sc, space=None, path=None):
    """Évalue les 512 profils (ou relit le CSV s'il existe déjà)."""
    space = space or DesignSpace()
    if path and os.path.exists(path):
        rows = list(csv.DictReader(open(path)))
        conv = {"valid": lambda v: v == "True", "feasible": lambda v: v == "True", "bits": str}
        out = []
        for r in rows:
            d = {}
            for k, v in r.items():
                if k in conv:
                    d[k] = conv[k](v)
                elif k in ("im", "ip", "it"):
                    d[k] = int(v)
                else:
                    d[k] = float(v)
            out.append(d)
        return out
    rows = [simulate(idx, space, sc) for idx in space.all_indices()]
    if path:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return rows


class CountedProblem:
    """Accès compté à la table : chaque profil distinct n'est compté qu'une fois."""

    def __init__(self, table, space=None, keys=("im", "ip", "it")):
        # keys : colonnes qui forment l'indice d'un profil ; V2 ajoute ("iI",) pour le rayon de BA
        self.space = space or DesignSpace()
        self.table = {tuple(r[k] for k in keys): r for r in table}
        self.seen = []          # ordre des simulations distinctes
        self._set = set()

    def __call__(self, idx):
        idx = tuple(int(i) for i in idx)
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
