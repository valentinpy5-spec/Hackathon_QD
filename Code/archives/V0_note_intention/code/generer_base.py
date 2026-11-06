"""Balayage exhaustif de l'espace de conception (512 profils par défaut).

Cette base sert trois usages :
  1. vérité terrain aéro pour calculer le front de Pareto EXACT (avec la SER
     de la phase 2), auquel NSGA-II et la boucle QUBO seront comparés ;
  2. tables de consultation rapides pendant le développement des phases 3-4
     (on ne compte alors les simulations qu'au travers d'AeroSimulator) ;
  3. diagnostic : régularité de CD(m, p, t), zones décollées, zones infaisables.

    python3 generer_base.py            # mode alpha imposé (FlightCase par défaut)
    python3 generer_base.py cl         # mode portance imposée
"""
import csv
import os
import sys
import time

from aero.design_space import DesignSpace, FlightCase, AeroSimulator

OUT = os.path.join(os.path.dirname(__file__), "resultats")
os.makedirs(OUT, exist_ok=True)

case = FlightCase(mode=sys.argv[1] if len(sys.argv) > 1 else "alpha")
sim = AeroSimulator(DesignSpace(), case)

t0 = time.time()
rows = []
for idx in sim.space.all_indices():
    r = sim(idx)
    d = r.to_dict()
    d.update(im=idx[0], ip=idx[1], it=idx[2],
             bits="".join(map(str, sim.space.encode(idx))),
             g=";".join(f"{g:.4f}" for g in sim.constraints(r)))
    rows.append(d)
dt = time.time() - t0

tag = f"Re{case.Re:.0e}_" + (f"a{case.alpha:g}" if case.mode == "alpha" else f"CL{case.CL_target:g}")
path = os.path.join(OUT, f"base_aero_{tag}.csv")
with open(path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

n_valid = sum(r["valid"] for r in rows)
n_feas = sum(all(float(g) <= 0 for g in r["g"].split(";")) for r in rows)
print(f"{len(rows)} profils en {dt:.1f} s ({1e3 * dt / len(rows):.0f} ms/profil)")
print(f"valides (pas de décollement avant x_sep_crit) : {n_valid}   faisables (toutes contraintes) : {n_feas}")
print(f"-> {path}")
