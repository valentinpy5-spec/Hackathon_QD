"""Export de la géométrie vers le modèle SER (interface phase 1 -> phase 2).

Le profil est extrudé selon l'envergure z pour former une maquette « 2,5D »
facettée en triangles. On réutilise EXACTEMENT les points du calcul de
panneaux : l'aéro et l'électromagnétisme voient la même forme discrète.

Conventions :
    - repère profil (corps) : x vers l'aval le long de la corde, y vers
      l'extrados, z selon l'envergure, origine au bord d'attaque ;
    - normales sortantes (ordre des sommets anti-horaire vu de l'extérieur) ;
    - l'incidence N'EST PAS appliquée à la géométrie : elle est écrite dans
      les métadonnées et c'est la direction radar qui est tournée.
"""
import json

import numpy as np


def extrude(X, Y, span=1.0, caps=True):
    """Triangles (n_tri, 3, 3) de la maquette extrudée.

    Chaque panneau 2D devient un rectangle de deux triangles. Les flancs
    (caps) sont triangulés en bandes entre points extrados et intrados de
    même indice, ce que permet la construction NACA (mêmes x paramétriques).
    """
    P0 = np.column_stack([X, Y, np.zeros_like(X)])
    P1 = P0 + np.array([0.0, 0.0, span])
    tris = []
    for k in range(len(X) - 1):
        A, B, C, D = P0[k], P0[k + 1], P1[k + 1], P1[k]
        tris += [(A, B, C), (A, C, D)]
    if caps:
        n_half = (len(X) - 1) // 2
        up = np.arange(n_half, -1, -1)        # BA -> BF extrados
        lo = np.arange(n_half, len(X))        # BA -> BF intrados
        for i in range(n_half):
            u0, u1, l0, l1 = up[i], up[i + 1], lo[i], lo[i + 1]
            # flanc z = 0, normale -z ; flanc z = span, normale +z
            for P, sgn in ((P0, -1), (P1, +1)):
                t1, t2 = (P[u0], P[l0], P[l1]), (P[u0], P[l1], P[u1])
                if sgn < 0:
                    t1, t2 = (t1[0], t1[2], t1[1]), (t2[0], t2[2], t2[1])
                tris += [t1, t2]
    T = np.array(tris)
    # retire les triangles dégénérés (bord d'attaque / bord de fuite fermé)
    area = 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
    return T[area > 1e-14]


def normals(T):
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def write_stl(T, path, name="naca"):
    """STL ASCII, lisible par la plupart des solveurs EM et par ParaView."""
    N = normals(T)
    with open(path, "w") as f:
        f.write(f"solid {name}\n")
        for n, tri in zip(N, T):
            f.write(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n    outer loop\n")
            for v in tri:
                f.write(f"      vertex {v[0]:.8e} {v[1]:.8e} {v[2]:.8e}\n")
            f.write("    endloop\n  endfacet\n")
        f.write(f"endsolid {name}\n")


def write_metadata(path, result, chord_m=1.0, span=1.0, n_tri=None):
    """Métadonnées associées au STL : design, incidence, échelle."""
    meta = dict(m=result.m, p=result.p, t=result.t, alpha_deg=result.alpha,
                chord_m=chord_m, span_chords=span, n_triangles=n_tri,
                frame="x: corde vers l'aval, y: extrados, z: envergure, origine au BA",
                radar_note="radar de face : direction d'incidence = -x tournée de alpha "
                           "autour de z (vent relatif)")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
