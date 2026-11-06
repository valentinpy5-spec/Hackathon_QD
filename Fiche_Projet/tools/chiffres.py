"""Écrit chiffres_auto.tex : les nombres de la fiche projet, lus dans les résultats du code.

Usage (depuis Fiche_Projet/) : python3 tools/chiffres.py, puis recompiler la fiche.
Chaque nombre est une macro LaTeX (\\VunSunNsga, ...) : quand une expérience est relancée,
la fiche se met à jour sans retoucher le texte.
"""
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "..", "Code", "resultats")
out = {}


def fr(x, nd=0):
    s = f"{x:.{nd}f}"
    return s.replace(".", "{,}")


# ---------------------------------------------------------------- V1 : chaîne complète
p = os.path.join(RES, "pipeline.json")
if os.path.exists(p):
    D = json.load(open(p))
    for sc, tag in (("S1_note", "Sun"), ("S2_sol", "Sdeux")):
        if sc not in D:
            continue
        S = D[sc]["summary"]
        out[f"Vun{tag}Front"] = str(D[sc]["n_front"])
        for k, name in (("nsga2", "Nsga"), ("random", "Hasard"), ("loop_exact", "Exact"),
                        ("loop_sa", "Recuit"), ("loop_qaoa", "Qaoa")):
            if k in S:
                out[f"Vun{tag}{name}"] = fr(S[k]["full_front"][1], 1 if S[k]["full_front"][1] % 1 else 0)
                out[f"Vun{tag}{name}Hv"] = fr(S[k]["hv99"][1], 1 if S[k]["hv99"][1] % 1 else 0)

# ---------------------------------------------------------------- V1 : validation XFOIL
px = os.path.join(RES, "base_xfoil_a2.csv")
po = os.path.join(RES, "base_aero_Re3e+06_a2.csv")
if os.path.exists(px) and os.path.exists(po):
    xf = {(r["im"], r["ip"], r["it"]): r for r in csv.DictReader(open(px))}
    ours = {(r["im"], r["ip"], r["it"]): r for r in csv.DictReader(open(po))}
    keys = [k for k in ours if xf[k]["xf_ok"] == "True" and ours[k]["valid"] == "True"]
    a = np.array([float(ours[k]["CD"]) for k in keys])
    b = np.array([float(xf[k]["xf_CD"]) for k in keys])
    rk = lambda v: np.argsort(np.argsort(v))
    out["VunXfConv"] = str(sum(r["xf_ok"] == "True" for r in xf.values()))
    out["VunXfBiaisCD"] = fr(100 * np.mean(a / b - 1))
    out["VunXfRangCD"] = fr(np.corrcoef(rk(a), rk(b))[0, 1], 2)

# ---------------------------------------------------------------- V2 : problème reformulé
p2 = os.path.join(RES, "pipeline_V2.json")
if os.path.exists(p2):
    D2 = json.load(open(p2))
    for sc, tag in (("S1", "Sun"), ("S2", "Sdeux")):
        S = D2[sc]["summary"]
        out[f"Vdeux{tag}Front"] = str(D2[sc]["n_front"])
        for k, name in (("random", "Hasard"), ("nsga2_8", "NsgaHuit"), ("nsga2_16", "Nsga"), ("loop_exact", "Exact"),
                        ("loop_sa", "Recuit"), ("loop_qaoa", "Qaoa")):
            med = S[k]["full_front"][1]
            out[f"Vdeux{tag}{name}"] = "---" if not np.isfinite(med) else fr(med, 1 if med % 1 else 0)
            out[f"Vdeux{tag}{name}Succes"] = fr(100 * S[k]["success"])
pm = os.path.join(RES, "montee_en_taille.json")
if os.path.exists(pm):
    M = json.load(open(pm))
    last = M[-1]
    out["VdeuxMonteeN"] = str(last["n"])
    out["VdeuxMonteeTopQuatre"] = fr(last["qaoa"]["4"]["p_top1"], 2)
    out["VdeuxMonteeTopDeux"] = fr(last["qaoa"]["2"]["p_top1"], 2)

with open(os.path.join(HERE, "chiffres_auto.tex"), "w") as f:
    f.write("% généré par tools/chiffres.py : ne pas modifier à la main\n")
    for k, v in out.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print("\n".join(f"{k} = {v}" for k, v in out.items()))
