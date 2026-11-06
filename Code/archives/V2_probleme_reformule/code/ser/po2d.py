"""SER simplifiée : optique physique (PO) 2D monostatique sur un contour polygonal.

Modèle provisoire de la phase 2, en attendant le solveur électromagnétique de
l'autre groupe. Il respecte l'interface prévue (même géométrie que l'aéro,
radar monostatique très haute fréquence, approche de face) pour que la chaîne
d'optimisation puisse être construite et testée dès maintenant.

Physique retenue (conducteur parfait, k c >> 1) :
    - courants de surface de l'optique physique J = 2 n x H_inc sur les
      facettes éclairées (n . s > 0), nuls dans l'ombre ;
    - largeur équivalente radar 2D (« echo width », par unité d'envergure) :
          sigma_2D = k | sum_{éclairées} (n_j . s) L_j sinc(k L_j t_j . s) e^{2 i k s . r_j} |^2
      où s est la direction unitaire vers le radar, r_j le milieu de la
      facette, t_j sa tangente, L_j sa longueur, sinc(x) = sin(x)/x.
      En 2D, la PO monostatique donne la même expression pour les deux
      polarisations (TM et TE).

Unités : longueurs en mètres (corde chord_m), sigma_2D en mètres ; en dB, on
écrit 10 log10(sigma_2D / 1 m) (dBm, « dB par mètre d'envergure »).
"""
import numpy as np

C0 = 299_792_458.0


def wavenumber(freq_hz):
    return 2 * np.pi * freq_hz / C0


def facets(X, Y):
    """Milieux, tangentes, normales sortantes et longueurs d'un contour fermé.

    Même convention que aero.panel : contour parcouru dans le sens trigonométrique
    (BF extrados -> BA -> BF intrados), normale sortante (t_y, -t_x).
    """
    dx, dy = np.diff(X), np.diff(Y)
    L = np.hypot(dx, dy)
    keep = L > 1e-14
    xs, ys = X[:-1][keep], Y[:-1][keep]
    dx, dy, L = dx[keep], dy[keep], L[keep]
    tx, ty = dx / L, dy / L
    return xs + dx / 2, ys + dy / 2, tx, ty, ty, -tx, L


def echo_width(X, Y, s_dirs, k):
    """sigma_2D (m) pour chaque direction vers le radar s_dirs (n_dir, 2)."""
    xc, yc, tx, ty, nx, ny, L = facets(X, Y)
    s = np.atleast_2d(s_dirs)
    ns = s[:, :1] * nx + s[:, 1:] * ny            # (n_dir, n_fac) : n . s
    ts = s[:, :1] * tx + s[:, 1:] * ty            # t . s
    rs = s[:, :1] * xc + s[:, 1:] * yc            # s . r
    lit = ns > 0
    amp = np.where(lit, ns * L * np.sinc(k * L * ts / np.pi) * np.exp(2j * k * rs), 0.0)
    return k * np.abs(amp.sum(axis=1)) ** 2


def radar_directions(alpha_deg, elev_deg):
    """Directions vers le radar, dans le repère du profil, pour une approche de face.

    Le profil vole à l'incidence alpha : dans son repère, le vent relatif vient
    de (cos a, sin a). Un radar placé devant l'avion, à l'élévation eps au-dessus
    de la trajectoire, est vu dans la direction s = -(cos(a - eps), sin(a - eps)).
    eps = 0 : radar exactement dans l'axe du vol.
    """
    ang = np.deg2rad(alpha_deg - np.asarray(elev_deg, dtype=float))
    return -np.column_stack([np.cos(ang), np.sin(ang)])


def to_db(sigma):
    return 10 * np.log10(np.maximum(sigma, 1e-30))


def sigma_3d_broadside(sigma_2d, span_m, k):
    """Passage 2D -> 3D pour un tronçon d'envergure b vu perpendiculairement à l'envergure :
    sigma_3D = 2 b^2 sigma_2D / lambda = k b^2 sigma_2D / pi (formule du cylindre fini)."""
    return k * span_m**2 * sigma_2d / np.pi
