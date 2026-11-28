"""Registre des affirmations du wrap-up : recalcule chaque chiffre depuis Code/resultats/.

Usage (depuis la racine du repo) : Code/.venv/Scripts/python.exe wrapup/registre.py
Sortie : wrapup/registre.csv (affirmation | valeur annoncée | valeur recalculée | source | commande | statut).
Aucun fichier de Code/resultats/ n'est modifié.
"""
import csv
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr, wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "Code", "resultats")
sys.path.insert(0, os.path.join(ROOT, "Code"))
CMD = "python wrapup/registre.py"
rows = []


def load(name):
    with open(os.path.join(RES, name), encoding="utf-8") as f:
        return json.load(f)


def add(claim, announced, value, source, tol=0.051, cmd=CMD, note=""):
    """statut : reproduit si |annoncé - recalculé| <= tol (absolu), écart sinon, non vérifiable si None."""
    if value is None:
        status = "non vérifiable"
    elif announced is None:
        status = "recalculé"
    else:
        status = "reproduit" if abs(float(announced) - float(value)) <= tol else "écart"
    v = value if value is None or isinstance(value, str) else round(float(value), 4)
    rows.append(dict(affirmation=claim, annonce=announced, recalcule=v, source=source,
                     commande=cmd, statut=status, note=note))


# ------------------------------------------------------------------ V3 L1 : traces brutes par graine
L1 = load("pipeline_L1_V3.json")


def full_front(trace):
    """Nombre de simulations au premier point où tout le front exact est trouvé (None si jamais)."""
    for n, frac, _hv in trace:
        if frac >= 1.0:
            return n
    return None


def sims(sec, m):
    return [full_front(t) for t in L1[sec][m]]


ANN = {  # valeurs annoncées (fiche, cours 5, PR #2)
    "S1": dict(loop_exact=75, quad_mean=75, gp_mean=87.5, quad_ei=85, gp_ei=125, parego=157.5),
    "S2": dict(loop_exact=142.5, quad_mean=95, gp_mean=110, quad_ei=117.5, gp_ei=127.5, parego=170),
}
PANN = {"S1": dict(quad_mean=0.35, gp_mean=0.18, quad_ei=0.11, gp_ei=0.001, parego=0.001),
        "S2": dict(quad_mean=0.001, gp_mean=0.002, quad_ei=0.03, gp_ei=0.06, parego=0.07)}
L1med = {}
for sec in ("S1", "S2"):
    base = sims(sec, "loop_exact")
    for m, a in ANN[sec].items():
        s = sims(sec, m)
        ok = [x for x in s if x is not None]
        # échec = +inf, comme dans le cours : la médiane reste définie si plus de la moitié réussit
        med = float(np.median([np.inf if x is None else x for x in s])) if 2 * len(ok) > len(s) else None
        L1med[(sec, m)] = med
        add(f"V3 L1 {sec} : {m}, simulations médianes pour le front exact complet (20 graines)",
            a, med, f"Code/resultats/pipeline_L1_V3.json:{sec}.{m}",
            note=f"succès {len(ok)}/{len(s)}")
        if m != "loop_exact" and len(ok) == len(s):
            p = wilcoxon(base, s).pvalue
            add(f"V3 L1 {sec} : Wilcoxon apparié boucle QUBO contre {m}, p", PANN[sec][m], p,
                f"pipeline_L1_V3.json:{sec} (cours 5, ch06_resultats.tex:39-47)", tol=0.02,
                note="p<0,001 annoncé noté 0,001 ; tolérance 0,02")

# ------------------------------------------------------------------ V2 et V1 : résumés
V2 = load("pipeline_V2.json")
S1v2 = V2["S1"]["summary"]
add("V2 S1 : hasard, simulations médianes pour le front exact", 911, S1v2["random"]["full_front"][1],
    "pipeline_V2.json:S1.summary.random")
add("V2 S1 : NSGA-II (N=16), taux de succès (%)", 5, 100 * S1v2["nsga2_16"]["success"],
    "pipeline_V2.json:S1.summary.nsga2_16", note="le pitch dit « fails in 95 % of runs »")
add("V2 S1 : boucle QAOA, taux de succès (%)", 100, 100 * S1v2["loop_qaoa"]["success"],
    "pipeline_V2.json:S1.summary.loop_qaoa", note="5 graines (cours 4, ch14:54)")
add("V2 S1 : boucle QAOA, simulations médianes", 105, S1v2["loop_qaoa"]["full_front"][1],
    "pipeline_V2.json:S1.summary.loop_qaoa", note="contre 75 pour le solveur exact")
add("V2 S2 : boucle QAOA, taux de succès (%)", 60, 100 * V2["S2"]["summary"]["loop_qaoa"]["success"],
    "pipeline_V2.json:S2.summary.loop_qaoa")
V1 = load("pipeline.json")
for k, a_exact, a_nsga in (("S1_note", 35, 106.5), ("S2_sol", 67.5, 106)):
    sm = V1[k]["summary"]
    add(f"V1 {k} : boucle QUBO exacte, simulations médianes", a_exact, sm["loop_exact"]["full_front"][1],
        f"pipeline.json:{k}.summary.loop_exact")
    add(f"V1 {k} : NSGA-II, simulations médianes", a_nsga, sm["nsga2"]["full_front"][1],
        f"pipeline.json:{k}.summary.nsga2")

# ------------------------------------------------------------------ ratios du pitch
add("Pitch « 12× fewer simulations » (hasard / boucle, V2-V3 S1)", 12, 911 / L1med[("S1", "loop_exact")],
    "911 / 75", tol=0.2)
add("Ratio réel contre l'optimisation bayésienne, S1", None, L1med[("S1", "gp_ei")] / L1med[("S1", "loop_exact")],
    "125 / 75")
add("Ratio contre l'optimisation bayésienne, S2 (< 1 : la boucle perd)", None,
    L1med[("S2", "gp_ei")] / L1med[("S2", "loop_exact")], "127,5 / 142,5")

# ------------------------------------------------------------------ L6 : CST 30 bits, campagne filtrée
E = load("ia_qubo_echelle.json")
B = load("ia_qubo_echelle_brut.json")
ref = E["hv_ref"]
for k, a_raw in (("quadz_modele", 0.882), ("fm_modele", 0.895), ("bocs_modele", 0.897),
                 ("hybride_modele", 0.913), ("nsga2_16", 0.812), ("hasard", 0.771)):
    med = float(np.median(E[k]["hv_final"]))
    add(f"L6 CST 30 bits : {k}, hypervolume BRUT médian à 300 simulations", a_raw, med,
        f"ia_qubo_echelle.json:{k}.hv_final", tol=0.006)
    add(f"L6 CST 30 bits : {k}, hypervolume RELATIF au front de référence (hv_ref={ref:.3f})", None, med / ref,
        "ia_qubo_echelle.json", note="le pitch présente le brut comme une fraction")


# Tests par graine : recalculés par Code/ia_qubo/audit_echelle.py (relancé dans une copie, sortie gardée ici)
AUD_CMD = "cd Code && python -m ia_qubo.audit_echelle (copie isolée ; sortie dans wrapup/reruns/)"
with open(os.path.join(ROOT, "wrapup", "reruns", "ia_qubo_audit_echelle.json"), encoding="utf-8") as f:
    AUD = json.load(f)
q100 = next(d for d in AUD["quadz_vs_nsga2"] if d["budget"] == 100)
add("L6 : boucle (modèle en z) contre NSGA-II à 100 simulations, Wilcoxon p", 0.008, q100["p"],
    "reruns/ia_qubo_audit_echelle.json:quadz_vs_nsga2", tol=0.001, cmd=AUD_CMD,
    note=f"{q100['gagne']}/8 graines ; p minimal possible à n=8 ; meilleur p sur 15 budgets")
add("L6 : même test après correction de Holm (non significatif à 5 %)", None,
    AUD["holm"]["quadz>nsga2_16@100"], "reruns/ia_qubo_audit_echelle.json:holm", cmd=AUD_CMD)
add("L6 : boucle contre NSGA-II à 300 simulations, Wilcoxon p (brut / Holm)", None,
    AUD["wilcoxon"]["nsga2_16@300"], "reruns/ia_qubo_audit_echelle.json", cmd=AUD_CMD,
    note=f"Holm {AUD['holm']['nsga2_16@300']:.3f}")
add("L6 : témoin hybride contre modèle en z à 300 simulations, Wilcoxon p", 0.11,
    AUD["wilcoxon"]["hybride_modele@300"], "reruns/ia_qubo_audit_echelle.json", tol=0.01, cmd=AUD_CMD,
    note="IC 90 % de l'écart hybride - z : " + str([round(x, 3) for x in AUD["ic90_ecart_moyen_vs_quadz_300"]["hybride_modele"]]))
for k in ("quadz_modele", "hybride_modele", "nsga2_16", "hasard"):
    add(f"L6 : {k}, hypervolume normalisé sur le front de référence, à 300 simulations", None,
        AUD["mediane"]["front"][k][1], "reruns/ia_qubo_audit_echelle.json:mediane.front", cmd=AUD_CMD,
        note="normalisation la plus discriminante (cours 6, ch09:124-138)")

# ------------------------------------------------------------------ QAOA, XFOIL
A = load("anatomie_qubo_V3.json")
add("Pénalités : part de l'étendue des énergies du QUBO (%)", 98, 100 * (1 - A["spectre"]["obj_share"]),
    "anatomie_qubo_V3.json:spectre.obj_share", tol=0.6)
po = [r["p_opt"] / r["p_random"] for k, v in A["qaoa"].items() for r in v["complet"]]
add("QAOA avec contraintes : p_opt / hasard (médiane sur les QUBO de la boucle)", 5, float(np.median(po)),
    "anatomie_qubo_V3.json:qaoa.*.complet", tol=1.5, note="cours 4 ch14 : « 5 fois le hasard »")
s = A["xfoil_s_par_profil"]
add("XFOIL : secondes par profil", 0.34, s, "anatomie_qubo_V3.json:xfoil_s_par_profil", tol=0.005)
add("10⁹ profils × XFOIL, en années", 10.8, 1e9 * s / 3.156e7, "calcul", tol=0.1,
    note=f"avec 2^30 profils exactement : {2**30 * s / 3.156e7:.1f} ans")
add("16 384 profils × XFOIL, en minutes", 93, 16384 * s / 60, "calcul", tol=1.5,
    note="le pitch demande des « compute hours » pour cette grille")

# ------------------------------------------------------------------ SER : gain du bord d'attaque pointu (I=6 -> I=2)
base = {}
with open(os.path.join(RES, "base_V2.csv"), encoding="utf-8") as f:
    for r in csv.DictReader(f):
        base[(r["im"], r["ip"], r["it"], r["iI"])] = (float(r["t"]), float(r["I"]))
ser = {}
with open(os.path.join(RES, "ser_mom_V2.csv"), encoding="utf-8") as f:
    for r in csv.DictReader(f):
        t, I = base[(r["im"], r["ip"], r["it"], r["iI"])]
        ser.setdefault((r["im"], r["ip"], round(t, 3)), {})[I] = (float(r["SER_S1_PO"]), float(r["SER_S1_MoM"]))
for t, a_po, a_mom in ((0.12, -7.3, -5.1), (0.08, -7.4, -4.0)):
    g_po, g_mom = [], []
    for (im, ip, tt), d in ser.items():
        if tt == t and 2.0 in d and 6.0 in d:
            g_po.append(d[2.0][0] - d[6.0][0])
            g_mom.append(d[2.0][1] - d[6.0][1])
    add(f"SER S1, t={t:.0%} : gain I=6→I=2 par optique physique (dB, médiane sur m,p)", a_po,
        float(np.median(g_po)) if g_po else None, "ser_mom_V2.csv + base_V2.csv", tol=0.6,
        note="cours 2 ch08_v2.tex:47")
    add(f"SER S1, t={t:.0%} : gain I=6→I=2 par méthode des moments, exacte (dB)", a_mom,
        float(np.median(g_mom)) if g_mom else None, "ser_mom_V2.csv + base_V2.csv", tol=0.6,
        note="le pitch annonce −7,3 dB « by method of moments »")

# ------------------------------------------------------------------ SER : méthode des moments contre Mie (recalcul)
try:
    from ser.mom2d import mie_tm, solve_tm
    errs = []
    lam = 0.03
    k = 2 * np.pi / lam
    for ka in (0.5, 2.0, 8.0, 31.0):
        a = ka / k
        n = max(64, int(15 * 2 * np.pi * a / lam))
        th = np.linspace(0, 2 * np.pi, n + 1)
        X, Y = a * np.cos(th), a * np.sin(th)
        w = solve_tm(X, Y, k, np.array([[1.0, 0.0]]))
        errs.append(abs(10 * np.log10(float(np.ravel(w)[0]) / mie_tm(ka, a))))
    add("MoM 2D TM contre série de Mie (cylindre), écart max en dB, ka = 0,5 à 31", 0.02, max(errs),
        "Code/ser/mom2d.py (recalcul)", tol=0.05, note="cours 2 ch08_v2.tex:37")
except Exception as e:
    add("MoM 2D TM contre série de Mie", 0.02, None, "Code/ser/mom2d.py", note=f"non recalculé : {e}")

# ------------------------------------------------------------------ aéro : modèle rapide contre XFOIL (grille 512)
xf, ours = {}, {}
with open(os.path.join(RES, "base_xfoil_a2.csv"), encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r["xf_ok"] == "True":
            xf[(r["im"], r["ip"], r["it"])] = (float(r["xf_CL"]), float(r["xf_CD"]))
with open(os.path.join(RES, "base_aero_Re3e+06_a2.csv"), encoding="utf-8") as f:
    for r in csv.DictReader(f):
        ours[(r["im"], r["ip"], r["it"])] = (float(r["CL"]), float(r["CD"]))
keys = [k for k in xf if k in ours]
cdx, cdo = np.array([xf[k][1] for k in keys]), np.array([ours[k][1] for k in keys])
clx, clo = np.array([xf[k][0] for k in keys]), np.array([ours[k][0] for k in keys])
add("XFOIL : profils convergés sur 512", 498, len(xf), "base_xfoil_a2.csv", tol=0)
add("Modèle rapide contre XFOIL : biais moyen sur CD (%)", 27, 100 * np.mean(cdo / cdx - 1), "base_*_a2.csv", tol=2)
add("Modèle rapide contre XFOIL : corrélation de rang sur CD", 0.91, spearmanr(cdo, cdx)[0], "base_*_a2.csv", tol=0.01)
add("Modèle rapide contre XFOIL : corrélation de rang sur CL", 0.94, spearmanr(clo, clx)[0], "base_*_a2.csv", tol=0.01)

# ------------------------------------------------------------------ V0, V1, PO : chiffres cités dans le deck
with open(os.path.join(ROOT, "Code", "archives", "V0_note_intention", "resultats", "experiences_quantique.json"),
          encoding="utf-8") as f:
    V0 = json.load(f)
add("V0 : boucle sur la traînée seule, simulations médianes jusqu'au vrai minimum", 25, V0["boucle"]["median_sims"],
    "archives/V0_note_intention/resultats/experiences_quantique.json:boucle.median_sims", note="fiche : valeur en dur")
add("V0 : hasard, simulations attendues", 256, V0["boucle"]["random_expected"],
    "experiences_quantique.json:boucle.random_expected", tol=0.6)
add("V0 : QAOA p=1, probabilité de l'optimum (9 qubits)", 0.018, V0["aero"]["qaoa"][0]["prob_opt"],
    "experiences_quantique.json:aero.qaoa", tol=0.001)
add("V0 : QAOA p=5, probabilité de l'optimum", 0.142, V0["aero"]["qaoa"][-1]["prob_opt"],
    "experiences_quantique.json:aero.qaoa", tol=0.001)
R1 = [r for r in csv.DictReader(open(os.path.join(RES, "base_complete_S1_note.csv"), encoding="utf-8"))
      if r["feasible"] == "True"]
F1 = np.array([[float(r["CD_counts"]), float(r["SER_db"])] for r in R1])
fr1 = [i for i, x in enumerate(F1) if not np.any(np.all(F1 <= x, axis=1) & np.any(F1 < x, axis=1))]
add("V1 S1 : profils sur le front exact", 7, len(fr1), "base_complete_S1_note.csv", tol=0)
add("V1 S1 : étendue de SER le long du front (dB)", 0.13, F1[fr1, 1].max() - F1[fr1, 1].min(),
    "base_complete_S1_note.csv", tol=0.01, note="épaisseurs du front : " + str(sorted({float(R1[i]["t"]) for i in fr1})))
poS = np.array([float(r["SER_S1_PO"]) for r in csv.DictReader(open(os.path.join(RES, "ser_mom_V2.csv"), encoding="utf-8"))])
moS = np.array([float(r["SER_S1_MoM"]) for r in csv.DictReader(open(os.path.join(RES, "ser_mom_V2.csv"), encoding="utf-8"))])
add("PO contre MoM, 1 024 profils S1 : écart moyen (dB)", -1.05, float(np.mean(poS - moS)), "ser_mom_V2.csv", tol=0.01)
add("PO contre MoM : corrélation de rang", 0.999, spearmanr(poS, moS)[0], "ser_mom_V2.csv", tol=0.001)

B2 = list(csv.DictReader(open(os.path.join(RES, "base_V2.csv"), encoding="utf-8")))
for I, a in (("2.0", 1.01), ("6.0", 1.61)):
    v = [float(r["CLmax"]) for r in B2 if r["I"] == I and r["t"] == "0.12" and r["CLmax"] != "nan"]
    add(f"V2, t=12 %, nez I={I[0]} : CL,max médian (XFOIL)", a, float(np.median(v)), "base_V2.csv", tol=0.006)
for I, a in (("2.0", 32), ("6.0", 84)):
    add(f"V2, nez I={I[0]} : part des profils faisables (%)", a,
        100 * np.mean([r["feasible"] == "True" for r in B2 if r["I"] == I]), "base_V2.csv", tol=0.6)

# ------------------------------------------------------------------ relances (journaux dans wrapup/reruns/)
add("V3 L1 relancé (5 méthodes × 2 scénarios × 20 graines) : traces identiques au JSON publié", None, "oui",
    "reruns/experiences_l1.log", cmd="cd Code && python -m optim.experiences_l1 S1 S2 loop_exact quad_mean gp_mean quad_ei gp_ei")
add("L6 audit relancé : hypervolumes recalculés depuis le cache égaux aux publiés (assert)", None, "oui",
    "reruns/ia_qubo_audit_echelle.json", cmd=AUD_CMD)
VAL = "cd Code && python validation.py (journal : wrapup/reruns/validation.log)"
add("Panneaux : pente dCL/dalpha NACA 0012 (/rad)", 6.89, 6.900, "reruns/validation.log", tol=0.011, cmd=VAL)
add("Thwaites contre Blasius sur plaque plane : écart max sur theta (%)", 1.0, 1.02, "reruns/validation.log", tol=0.1, cmd=VAL)
add("Head contre loi en 1/7 : écart max sur theta (%)", None, 4.9, "reruns/validation.log", cmd=VAL)
add("Comparaison XFOIL (validation V5, grille XFOIL)", None, None, "validation.py, validation_xfoil_grille.py",
    cmd=VAL, note="XFOIL absent sur cette machine (Windows) : non relancé ; résumé recalculé depuis base_xfoil_a2.csv")

# ------------------------------------------------------------------ sortie
out = os.path.join(ROOT, "wrapup", "registre.csv")
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
n = {s: sum(r["statut"] == s for r in rows) for s in ("reproduit", "écart", "recalculé", "non vérifiable")}
print(f"{len(rows)} lignes -> {out} ; {n}")
for r in rows:
    if r["statut"] in ("écart", "non vérifiable"):
        print(f"  [{r['statut']}] {r['affirmation']} : annoncé {r['annonce']}, recalculé {r['recalcule']} {r['note']}")
