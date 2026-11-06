"""Vérifie que la chaîne V2 tourne sur cette machine, en moins d'une minute, sans toucher à resultats/.

    python3 verifier_installation.py

1. un profil complet : XFOIL (croisière et manœuvre) puis SER par optique physique ;
2. SER par méthode des moments sur le même profil ;
3. une exécution de la boucle QUBO (solveur exact) sur la table V2 en cache, scénario S1.
"""
import time

import numpy as np

t0 = time.time()
print(f"NumPy {np.__version__}")

from aero.xfoil_runner import find_xfoil  # noqa: E402

exe = find_xfoil()
print(f"XFOIL : {exe or 'INTROUVABLE (lancer ./installer_outils.sh ou installer xfoil)'}")
if exe is None:
    raise SystemExit(1)

from optim.problem_v2 import simulate_v2  # noqa: E402

idx = (2, 2, 3, 2)                                     # m 2 %, p 0,45, t 12 %, I = 6
r = simulate_v2(idx)
print(f"1. profil m={r['m']:.2f} p={r['p']:.2f} t={r['t']:.2f} I={r['I']:g} : "
      f"CD = {r['CD_counts']:.1f} counts, alpha = {r['alpha']:.2f}°, CLmax = {r['CLmax']:.2f}, "
      f"SER S1 (optique physique) = {r['SER_S1']:.2f} dBm")
if not r["valid"]:
    raise SystemExit("XFOIL n'a pas convergé : vérifier l'installation (affichage, Xvfb)")

from ser.compare_po_mom import ser_both  # noqa: E402

po, mo, n = ser_both(r["m"], r["p"], r["t"], r["I"], r["alpha"], -20.0, 20.0, n=21)
print(f"2. SER S1 : optique physique {po:.2f} dBm, méthode des moments {mo:.2f} dBm ({n} segments)")

from optim.experiences_v2 import setup, run  # noqa: E402

T, feas, F, fr, exact, norm, ref, hv = setup("S1", "MoM")
tr = run("loop_exact", T, exact, norm, ref, hv, seeds=[0])[0]
n_full = next((row[0] for row in tr if row[1] >= 1.0), None)
print(f"3. boucle QUBO, solveur exact, S1 : front de {len(fr)} profils retrouvé en {n_full} simulations")
print(f"Installation correcte ({time.time() - t0:.0f} s).")
