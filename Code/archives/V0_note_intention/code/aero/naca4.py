"""Géométrie NACA 4 chiffres.

Paramètres (fractions de corde) :
    m : cambrure maximale          (NACA 2412 -> m = 0.02)
    p : position de la cambrure max (NACA 2412 -> p = 0.4)
    t : épaisseur relative max      (NACA 2412 -> t = 0.12)

Convention de parcours (identique à XFOIL) : bord de fuite extrados -> bord
d'attaque -> bord de fuite intrados. Le contour est donc parcouru dans le sens
trigonométrique, la normale sortante d'un segment de tangente (tx, ty) est
(ty, -tx).
"""
import numpy as np


def camber_line(x, m, p):
    """Ligne moyenne yc(x) et sa pente dyc/dx."""
    x = np.asarray(x, dtype=float)
    yc = np.zeros_like(x)
    dyc = np.zeros_like(x)
    if m == 0.0 or p == 0.0:
        return yc, dyc
    fwd = x < p
    yc[fwd] = m / p**2 * (2 * p * x[fwd] - x[fwd] ** 2)
    dyc[fwd] = 2 * m / p**2 * (p - x[fwd])
    aft = ~fwd
    yc[aft] = m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x[aft] - x[aft] ** 2)
    dyc[aft] = 2 * m / (1 - p) ** 2 * (p - x[aft])
    return yc, dyc


def thickness(x, t, closed_te=True):
    """Demi-épaisseur yt(x). closed_te ferme le bord de fuite (coef -0.1036)."""
    a4 = -0.1036 if closed_te else -0.1015
    return 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2
                    + 0.2843 * x**3 + a4 * x**4)


def naca4(m, p, t, n_half=100, closed_te=True):
    """Coordonnées du profil, 2*n_half + 1 points, distribution en cosinus.

    Le cosinus resserre les points au bord d'attaque et au bord de fuite, là où
    la courbure et les gradients de vitesse sont les plus forts.
    Renvoie (X, Y) avec X[0] = X[-1] = 1 (bord de fuite).
    """
    beta = np.linspace(0.0, np.pi, n_half + 1)
    x = 0.5 * (1 - np.cos(beta))
    yt = thickness(x, t, closed_te)
    yc, dyc = camber_line(x, m, p)
    th = np.arctan(dyc)
    xu, yu = x - yt * np.sin(th), yc + yt * np.cos(th)
    xl, yl = x + yt * np.sin(th), yc - yt * np.cos(th)
    X = np.concatenate([xu[::-1], xl[1:]])
    Y = np.concatenate([yu[::-1], yl[1:]])
    return X, Y


def naca4_from_code(code, **kw):
    """'2412' -> naca4(0.02, 0.4, 0.12)."""
    code = str(code).zfill(4)
    return naca4(int(code[0]) / 100, int(code[1]) / 10, int(code[2:]) / 100, **kw)


def leading_edge_radius(t):
    """Rayon de bord d'attaque NACA 4 chiffres : r = 1.1019 t^2 (en cordes).

    Grandeur clé pour le compromis aéro/SER : elle pilote à la fois le pic
    d'aspiration et la rétrodiffusion radar en approche de face.
    """
    return 1.1019 * t**2
