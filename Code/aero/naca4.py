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


# ============================================================================
# V2 : série NACA 4 chiffres MODIFIÉE (Abbott & von Doenhoff, §6.4 ; Ladson et
# Brooks, NASA TM X-3284, 1975). Elle ajoute deux paramètres à la loi
# d'épaisseur : l'indice de rayon de bord d'attaque I (I = 6 : rayon du NACA
# standard, r = 1.1019 t^2 (I/6)^2) et la position de l'épaisseur maximale T.
# C'est ce qui permet, en V2, de séparer le rayon de bord d'attaque (qui pilote
# la SER de face) de l'épaisseur (qui pilote la traînée et la structure).
# ============================================================================
def _modified_coeffs(I, T):
    """Coefficients de la loi d'épaisseur modifiée pour un profil de 20 % (demi-épaisseur 0.1 en x = T).

    Arrière (x >= T) : y = d1 (1-x) + d2 (1-x)^2 + d3 (1-x)^3   (bord de fuite fermé, d0 = 0)
    Avant  (x <= T)  : y = a0 sqrt(x) + a1 x + a2 x^2 + a3 x^3
    Conditions en x = T : y = 0.1, y' = 0, et courbure continue entre les deux arcs.
    """
    d1 = (2.24 - 5.42 * T + 12.3 * T**2) / (10.0 * (1.0 - 0.878 * T))   # pente au bord de fuite
    u = 1.0 - T
    d2, d3 = np.linalg.solve([[u**2, u**3], [2 * u, 3 * u**2]], [0.1 - d1 * u, -d1])
    ypp = 2 * d2 + 6 * d3 * u                                            # y'' de l'arc arrière en T
    a0 = 0.296904 * I / 6.0
    M = [[T, T**2, T**3], [1.0, 2 * T, 3 * T**2], [0.0, 2.0, 6 * T]]
    rhs = [0.1 - a0 * np.sqrt(T), -0.5 * a0 / np.sqrt(T), ypp + 0.25 * a0 * T**-1.5]
    a1, a2, a3 = np.linalg.solve(M, rhs)
    return a0, a1, a2, a3, d1, d2, d3


def thickness_modified(x, t, I=6.0, T=0.3):
    """Demi-épaisseur de la série modifiée, mise à l'échelle de l'épaisseur t."""
    x = np.asarray(x, dtype=float)
    a0, a1, a2, a3, d1, d2, d3 = _modified_coeffs(I, T)
    fwd = a0 * np.sqrt(x) + a1 * x + a2 * x**2 + a3 * x**3
    v = 1.0 - x
    aft = d1 * v + d2 * v**2 + d3 * v**3
    return (t / 0.2) * np.where(x <= T, fwd, aft)


def naca4_modified(m, p, t, I=6.0, T=0.3, n_half=100):
    """Contour (X, Y) d'un profil NACA 4 chiffres modifié, même convention que naca4."""
    beta = np.linspace(0.0, np.pi, n_half + 1)
    x = 0.5 * (1 - np.cos(beta))
    yt = thickness_modified(x, t, I, T)
    yc, dyc = camber_line(x, m, p)
    th = np.arctan(dyc)
    xu, yu = x - yt * np.sin(th), yc + yt * np.cos(th)
    xl, yl = x + yt * np.sin(th), yc - yt * np.cos(th)
    return np.concatenate([xu[::-1], xl[1:]]), np.concatenate([yu[::-1], yl[1:]])


def leading_edge_radius_modified(t, I):
    """r = 1.1019 (t I / 6)^2 : le rayon varie comme le carré de I, à épaisseur fixée."""
    return 1.1019 * (t * I / 6.0) ** 2
