"""Données de l'explorateur de profils (wrapup/explorateur.html), tirées de Code/resultats/.

Usage (depuis la racine du repo) : Code/.venv/Scripts/python.exe wrapup/explorateur.py
Sortie : wrapup/explorateur_data.json, lu par la page. Aucun fichier de Code/resultats/ n'est modifié.

Trois problèmes, chacun avec tous ses profils simulés, son front et de quoi redessiner chaque forme en JS :
    V1     512 NACA 4 chiffres, modèle rapide + SER par optique physique (base_complete_S*.csv) ;
    V2     1 024 NACA modifiés, XFOIL + SER PO ou MoM (base_V2.csv, ser_mom_V2.csv), trace de la boucle S2 ;
    CST    les 16 244 profils simulés par la campagne L6 filtrée, et l'ordre de simulation de chaque méthode.
"""
import csv
import json
import os
import sys
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "Code", "resultats")
sys.path.insert(0, os.path.join(ROOT, "Code"))
sys.modules.setdefault("fcntl", types.ModuleType("fcntl"))  # probleme_cst l'importe (verrou du cache) ; absent sous Windows, inutile ici

from aero.naca4 import _modified_coeffs  # noqa: E402
from ia_qubo import probleme_cst as P  # noqa: E402
from optim.problem_v2 import CaseV2, SpaceV2  # noqa: E402


def rows(name):
    with open(os.path.join(RES, name), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load(name):
    with open(os.path.join(RES, name), encoding="utf-8") as f:
        return json.load(f)


def num(v, nd=4):
    """Nombre JSON compact ; NaN -> None."""
    v = float(v)
    return None if not np.isfinite(v) else round(v, nd)


def pareto(F):
    """Indices non dominés (minimisation des deux colonnes), comme wrapup/build.py."""
    F = np.asarray(F)
    return [i for i, f in enumerate(F) if not np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1))]


def front_of(pts, ser_key):
    ok = [i for i, p in enumerate(pts) if p["f"] and p["cd"] is not None and p.get(ser_key) is not None]
    fr = pareto([[pts[i]["cd"], pts[i][ser_key]] for i in ok])
    return sorted((ok[j] for j in fr), key=lambda i: pts[i]["cd"])


# ------------------------------------------------------------------ V1 : NACA 4 chiffres, modèle rapide
def v1():
    s1, s2 = rows("base_complete_S1_note.csv"), rows("base_complete_S2_sol.csv")
    pts = []
    for a, b in zip(s1, s2):
        assert (a["im"], a["ip"], a["it"]) == (b["im"], b["ip"], b["it"]) and a["feasible"] == b["feasible"]
        pts.append(dict(id="".join((a["im"], a["ip"], a["it"])), m=num(a["m"]), p=num(a["p"]), t=num(a["t"]),
                        cd=num(a["CD_counts"], 2) if a["valid"] == "True" else None, cl=num(a["CL"]),
                        S1=num(a["SER_db"], 3), S2=num(b["SER_db"], 3), f=a["feasible"] == "True",
                        why=None if a["feasible"] == "True" else ("t" if float(a["t"]) < 0.08 else
                                                                   "cv" if a["valid"] != "True" else "cl")))
    fronts = {s: front_of(pts, s) for s in ("S1", "S2")}
    print(f"V1 : {len(pts)} profils, {sum(p['f'] for p in pts)} faisables, fronts S1 {len(fronts['S1'])}, S2 {len(fronts['S2'])}")
    return dict(label="V1 · chaîne de la note", n=len(pts), pts=pts, fronts=fronts, geom="naca4",
                sources=["base_complete_S1_note.csv", "base_complete_S2_sol.csv"])


# ------------------------------------------------------------------ V2 : NACA modifié, XFOIL, PO et MoM
def v2():
    mom = {(r["im"], r["ip"], r["it"], r["iI"]): r for r in rows("ser_mom_V2.csv")}
    case = CaseV2()
    pts = []
    for r in rows("base_V2.csv"):
        k = (r["im"], r["ip"], r["it"], r["iI"])
        ok = r["valid"] == "True"
        cl = float(r["CLmax"])
        why = None
        if r["feasible"] != "True":
            why = "t" if float(r["t"]) < case.t_min else "cv" if not ok else "cl"
        pts.append(dict(id="".join(k), m=num(r["m"]), p=num(r["p"]), t=num(r["t"]), I=num(r["I"]),
                        rle=num(r["r_le"], 6), al=num(r["alpha"], 3), cd=num(r["CD_counts"], 2) if ok else None,
                        clmax=num(cl), S1=num(mom[k]["SER_S1_MoM"], 3), S2=num(mom[k]["SER_S2_MoM"], 3),
                        S1po=num(r["SER_S1"], 3), S2po=num(r["SER_S2"], 3), f=r["feasible"] == "True", why=why))
    fronts = {s: front_of(pts, s) for s in ("S1", "S2", "S1po", "S2po")}
    pub = load("pipeline_V2.json")
    for s in ("S1", "S2"):  # mêmes fronts que la campagne publiée
        assert len(fronts[s]) == pub[s]["n_front"], (s, len(fronts[s]), pub[s]["n_front"])
        assert sum(p["f"] for p in pts) == pub[s]["n_feasible"]
    print(f"V2 : {len(pts)} profils, {sum(p['f'] for p in pts)} faisables, "
          + ", ".join(f"front {s} {len(v)}" for s, v in fronts.items()))

    T = load("trace_boucle_V2.json")
    assert T["sector"] == "S2"
    steps = [["".join(map(str, c["idx"])) for c in T["init"]]]
    steps += [["".join(map(str, c["idx"])) for c in it["cand"]] for it in T["iterations"]]
    replay = dict(label="Boucle QUBO, S2, graine 0", sector="S2", seed=T["seed"], steps=steps,
                  source="trace_boucle_V2.json")
    # loi d'épaisseur modifiée : coefficients par indice de nez I (T = 0,3 fixé en V2), pour redessiner en JS
    coeffs = {str(int(I)): [float(c) for c in _modified_coeffs(I, case.T)] for I in SpaceV2().I_levels}
    return dict(label="V2 · problème reformulé", n=len(pts), pts=pts, fronts=fronts, geom="naca4mod",
                coeffs=coeffs, T=case.T, replay=replay, sources=["base_V2.csv", "ser_mom_V2.csv", "pipeline_V2.json"])


# ------------------------------------------------------------------ CST : campagne L6 filtrée (30 bits)
def cst():
    E = load("ia_qubo_echelle.json")
    B = load("ia_qubo_echelle_brut.json")
    assert E["filtre"] and E["cd_aberrant"] == P.CD_ABERRANT
    cache = P._charge_cache()                      # applique le seuil de 200 counts, comme la campagne
    runs = {m: [[tuple(x) for x in seed[0]] for seed in B[m]] for m in B}
    simulated = sorted({x for m in runs for s in runs[m] for x in s})
    assert len(simulated) == E["n_simules_total"], (len(simulated), E["n_simules_total"])
    pts = []
    for x in simulated:
        r = cache[x]
        why = None
        if not r["feasible"]:
            why = ("geom" if not r["geom_ok"] else "t" if r["t_max"] < P.T_MIN else "cv" if not r["valid"]
                   else "cl" if not (r["CLmax"] >= P.CL_MAX_MIN) else "cv")
        pts.append(dict(id="".join(map(str, x)), t=num(r["t_max"]), al=num(r["alpha"], 3),
                        cd=num(r["CD_counts"], 1) if r["valid"] else None, clmax=num(r["CLmax"]),
                        S2=num(r["SER_db"], 3), f=bool(r["feasible"]), why=why))
    assert sum(p["f"] for p in pts) == E["n_faisables_total"]
    fronts = {"S2": front_of(pts, "S2")}
    ref = sorted("".join(map(str, x)) for x in E["front_ref"])
    assert sorted(pts[i]["id"] for i in fronts["S2"]) == ref, "front différent de front_ref publié"
    print(f"CST : {len(pts)} profils simulés, {sum(p['f'] for p in pts)} faisables, front {len(fronts['S2'])} (= front_ref)")
    labels = dict(hasard="Hasard", nsga2_16="NSGA-II", quadz_modele="Boucle QUBO (quadratique en z)",
                  fm_modele="Boucle QUBO (machine de factorisation)", fm_bn="Boucle QUBO (factorisation, contrainte en boîte noire)",
                  bocs_modele="Boucle QUBO (BOCS)", hybride_modele="Témoin hybride, sans QUBO")
    replay = {m: dict(label=labels[m], seeds=[["".join(map(str, x)) for x in s] for s in runs[m]]) for m in runs}
    vals = [[float(v) for v in lv] for lv in P.VALEURS]
    return dict(label="V3 L6 · profil CST, 30 bits", n=len(pts), pts=pts, fronts=fronts, geom="cst", vals=vals,
                nbern=P.N_BERN, replay=replay, sources=["cache_cst.jsonl", "ia_qubo_echelle.json",
                                                        "ia_qubo_echelle_brut.json"])


def compact(pb):
    """Colonnes plutôt qu'objets (fichier 2 fois plus léger) ; profils des rejeux remplacés par leur rang dans pts."""
    pts = pb.pop("pts")
    cols = list(pts[0])
    pb["cols"], pb["rows"] = cols, [[p[c] for c in cols] for p in pts]
    rank = {p["id"]: i for i, p in enumerate(pts)}
    rp = pb.get("replay")
    if rp and "steps" in rp:
        rp["steps"] = [[rank[x] for x in st] for st in rp["steps"]]
    elif rp:
        for m in rp.values():
            m["seeds"] = [[rank[x] for x in s] for s in m["seeds"]]
    return pb


if __name__ == "__main__":
    data = {k: compact(f()) for k, f in (("V1", v1), ("V2", v2), ("CST", cst))}
    out = os.path.join(ROOT, "wrapup", "explorateur_data.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    print(f"-> {out} ({os.path.getsize(out) / 1e3:.0f} ko)")
