"""V2 : SER 2D exacte par méthode des moments (équation intégrale du champ électrique, polarisation TM).

But : valider l'optique physique (po2d.py) là où elle est douteuse, en particulier sur
les nez aigus de la V2 (k r_BA = 0,16 à 0,37 à 10 GHz, bien hors du régime optique).

Polarisation TM : le champ électrique est parallèle à l'envergure (axe z). Sur un
conducteur parfait, le champ total tangent s'annule :
    E_z^inc(r) = (k eta / 4) ∫ J_z(r') H0^(2)(k |r - r'|) dl'      sur le contour.
Discrétisation : courant constant par segment, collocation au milieu (point matching).
    Z_mn = (k eta / 4) L_n H0^(2)(k R_mn)                           m != n
    Z_nn = (k eta / 4) L_n [1 - (2j/pi) (ln(gamma k L_n / 4) - 1)]  terme propre (gamma = 1,781)
Largeur équivalente monostatique (onde incidente d'amplitude 1 venant de s) :
    sigma_2D = (k eta^2 / 4) | sum_n J_n L_n exp(j k s . r_n) |^2

Limite connue : l'EFIE a des résonances internes parasites (fréquences propres de la
cavité intérieure). On le vérifie sur le cylindre, comparé à la série de Mie.
"""
import numpy as np
from scipy.special import hankel2

ETA = 376.730313668
GAMMA = 1.781072418


def resample(X, Y, h_max, refine=None):
    """Rééchantillonne le contour fermé avec des segments de longueur <= h_max.

    refine(x) <= 1 : facteur de raffinement local (par exemple près du nez)."""
    seg = np.hypot(np.diff(X), np.diff(Y))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    pts = [0.0]
    while pts[-1] < s[-1]:
        x = np.interp(pts[-1], s, X)
        h = h_max * (refine(x) if refine else 1.0)
        pts.append(min(pts[-1] + h, s[-1]))
    pts = np.array(pts)
    return np.interp(pts, s, X), np.interp(pts, s, Y)


def solve_tm(X, Y, k, s_dirs):
    """Courants et largeur équivalente TM pour plusieurs directions radar s (n_dir, 2)."""
    xs, ys, xe, ye = X[:-1], Y[:-1], X[1:], Y[1:]
    L = np.hypot(xe - xs, ye - ys)
    keep = L > 1e-12
    xs, ys, xe, ye, L = xs[keep], ys[keep], xe[keep], ye[keep], L[keep]
    xc, yc = 0.5 * (xs + xe), 0.5 * (ys + ye)
    R = np.hypot(xc[:, None] - xc[None, :], yc[:, None] - yc[None, :])
    np.fill_diagonal(R, 1.0)
    Z = (k * ETA / 4) * L[None, :] * hankel2(0, k * R)
    np.fill_diagonal(Z, (k * ETA / 4) * L * (1 - (2j / np.pi) * (np.log(GAMMA * k * L / 4) - 1)))
    s = np.atleast_2d(s_dirs)
    # onde incidente venant de la direction s : E_inc = exp(j k s . r)
    E = np.exp(1j * k * (xc[:, None] * s[:, 0] + yc[:, None] * s[:, 1]))
    J = np.linalg.solve(Z, E)
    far = (L[:, None] * J * E).sum(axis=0)          # réciprocité : même phase à l'émission et au retour
    return (k * ETA**2 / 4) * np.abs(far) ** 2


def mie_tm(ka, a):
    """Largeur équivalente monostatique exacte (TM) d'un cylindre conducteur de rayon a."""
    from scipy.special import jv
    N = int(ka + 10 * ka ** (1 / 3) + 15)
    n = np.arange(-N, N + 1)
    sm = np.sum((-1.0) ** n * jv(n, ka) / hankel2(n, ka))
    k = ka / a
    return 4 * abs(sm) ** 2 / k
