"""Point d'entrée de la phase 2 (modèle SER provisoire) : profil + attitude -> SER.

Interface volontairement identique à ce que fournira le solveur EM de l'autre
groupe : la géométrie est celle de la phase 1 (mêmes points NACA), l'attitude
est l'incidence alpha calculée par l'aéro, et la sortie principale est une SER
moyenne sur un secteur d'approche de face. Remplacer ce module par le vrai
solveur ne doit rien changer ailleurs.
"""
from dataclasses import dataclass, field

import numpy as np

from aero.naca4 import naca4, naca4_modified, leading_edge_radius, leading_edge_radius_modified
from .po2d import wavenumber, echo_width, radar_directions, to_db


@dataclass
class RadarCase:
    """Scénario radar commun à toutes les phases (à figer avec le groupe SER)."""
    freq_hz: float = 10e9       # bande X : « très haute fréquence » devant la corde
    chord_m: float = 1.0        # corde physique (fixe le rapport c / lambda)
    elev_max: float = 20.0      # secteur d'approche de face : élévation dans [-elev_max, +elev_max] (°)
    n_elev: int = 81            # nombre de directions dans le secteur
    n_half: int = 200           # discrétisation du contour pour la SER (convergée, voir cours)

    @property
    def k(self):
        return wavenumber(self.freq_hz)

    @property
    def elevations(self):
        return np.linspace(-self.elev_max, self.elev_max, self.n_elev)


@dataclass
class SERResult:
    m: float
    p: float
    t: float
    alpha: float
    ser_db: float          # objectif : 10 log10 de la moyenne (linéaire) de sigma_2D sur le secteur (dBm)
    ser_nose_db: float     # radar exactement dans l'axe du vol (eps = 0)
    ser_max_db: float      # pire direction du secteur
    r_le: float            # rayon de bord d'attaque (m)
    pattern: np.ndarray = field(default=None, repr=False)   # sigma_2D(eps) (m)

    def to_dict(self):
        return dict(ser_db=self.ser_db, ser_nose_db=self.ser_nose_db,
                    ser_max_db=self.ser_max_db, r_le_m=self.r_le)


def evaluate_ser(m, p, t, alpha=0.0, case=None, I=None, T=0.3):
    """I (indice de rayon de bord d'attaque, V2) : si donné, géométrie NACA 4 chiffres modifiée."""
    case = case or RadarCase()
    if I is None:
        X, Y = naca4(m, p, t, n_half=case.n_half)
    else:
        X, Y = naca4_modified(m, p, t, I, T, n_half=case.n_half)
    X, Y = X * case.chord_m, Y * case.chord_m
    eps = case.elevations
    sig = echo_width(X, Y, radar_directions(alpha, eps), case.k)
    nose = echo_width(X, Y, radar_directions(alpha, [0.0]), case.k)[0]
    return SERResult(m, p, t, alpha, float(to_db(sig.mean())), float(to_db(nose)),
                     float(to_db(sig.max())),
                     (leading_edge_radius(t) if I is None else leading_edge_radius_modified(t, I)) * case.chord_m, sig)
