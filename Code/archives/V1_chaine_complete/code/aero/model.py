"""Point d'entrée de la phase 1 : profil NACA (m, p, t) -> CL, CD, CM, ...

C'est la seule fonction que les autres phases appellent :
    - NSGA-II (pymoo) : objectif traînée + contrainte de portance ;
    - boucle QUBO     : « simulation » des profils candidats ;
    - modèle SER      : récupère la géométrie (mêmes points) et l'incidence.
"""
from dataclasses import dataclass, field, asdict

import numpy as np
from scipy.optimize import brentq

from .naca4 import naca4, leading_edge_radius
from .panel import solve
from .boundary_layer import march_surface, split_at_stagnation, squire_young


@dataclass
class AeroResult:
    m: float
    p: float
    t: float
    alpha: float            # degrés
    Re: float
    CL: float
    CD: float               # traînée de profil (Squire-Young, somme des deux faces)
    CD_f: float             # part de frottement (intégrale de cf), CD - CD_f = traînée de forme
    CM: float               # quart de corde, positif à cabrer
    CL_inv: float           # = CL (couplage faible : pas de correction visqueuse de CL)
    x_tr_up: float
    x_tr_lo: float
    tr_mode_up: str
    tr_mode_lo: str
    x_sep_up: float
    x_sep_lo: float
    r_le: float             # rayon de bord d'attaque (cordes)
    valid: bool             # False si décollement turbulent avant x_sep_crit ou échec numérique
    flags: list = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        d["flags"] = ";".join(self.flags)
        return d


def _friction_drag(bl, cos_wall):
    """Intégrale de cf (Ue/U)^2 cos(paroi) ds : frottement projeté sur la corde."""
    return np.sum(bl.cf * bl.Ue**2 * np.gradient(bl.s) * cos_wall)


def evaluate(m, p, t, alpha=0.0, Re=3e6, n_half=100, x_trip=(None, None),
             x_stop=0.985, free_transition=True,
             x_sep_crit=0.9, return_details=False):
    """Évalue un profil NACA 4 chiffres à incidence fixée.

    m, p, t : cambrure, position de cambrure, épaisseur (fractions de corde)
    alpha   : incidence en degrés
    Re      : Reynolds basé sur la corde
    x_trip  : (x_tr extrados, x_tr intrados) forcés, None = transition libre
    free_transition : False -> transition uniquement en x_trip (ou au décollement laminaire)
    x_stop  : x/c d'arrêt de la couche limite et d'application de Squire-Young
    x_sep_crit : un décollement turbulent en amont de cette abscisse invalide le calcul
    """
    X, Y = naca4(m, p, t, n_half=n_half)
    flags = []
    inv = solve(X, Y, alpha)
    try:
        up, lo = split_at_stagnation(inv)
    except RuntimeError as e:
        flags.append(str(e))
        res = AeroResult(m, p, t, alpha, Re, inv.CL, np.nan, np.nan, inv.CM, inv.CL,
                         np.nan, np.nan, "", "", np.nan, np.nan,
                         leading_edge_radius(t), False, flags)
        return (res, inv, None) if return_details else res

    bl_up = march_surface(*up[:3], Re, x_trip=x_trip[0], x_stop=x_stop,
                         free_transition=free_transition)
    bl_lo = march_surface(*lo[:3], Re, x_trip=x_trip[1], x_stop=x_stop,
                         free_transition=free_transition)
    flags += [f"up:{f}" for f in bl_up.flags] + [f"lo:{f}" for f in bl_lo.flags]

    CD = (squire_young(bl_up.theta_te, bl_up.H_te, bl_up.Ue_te)
          + squire_young(bl_lo.theta_te, bl_lo.H_te, bl_lo.Ue_te))
    # frottement : cos de l'angle entre la paroi et l'écoulement ~ |tx| (petites incidences)
    tx = np.abs(inv.panels.tx)
    CD_f = (_friction_drag(bl_up, tx[up[3][:bl_up.s.size]])
            + _friction_drag(bl_lo, tx[lo[3][:bl_lo.s.size]]))

    # Un décollement turbulent en aval de x_sep_crit est signalé mais toléré : le
    # gradient adverse du bord de fuite fermé y est exagéré par l'écoulement
    # potentiel, et CD reste plausible. En amont, le modèle sort de son domaine.
    x_sep = np.nanmin([bl_up.x_sep, bl_lo.x_sep, np.inf])
    if np.isfinite(x_sep) and x_sep >= x_sep_crit:
        flags.append("te_separation")
    valid = (x_sep >= x_sep_crit and np.isfinite(CD)
             and not any("failed" in f or "no_transition" in f for f in flags))
    res = AeroResult(m, p, t, alpha, Re, inv.CL, CD, CD_f, inv.CM, inv.CL,
                     bl_up.x_tr, bl_lo.x_tr, bl_up.tr_mode, bl_lo.tr_mode,
                     bl_up.x_sep, bl_lo.x_sep, leading_edge_radius(t), valid, flags)
    if return_details:
        return res, inv, (bl_up, bl_lo)
    return res


def alpha_for_cl(m, p, t, CL_target, bracket=(-10.0, 15.0), n_half=100):
    """Incidence donnant CL_target (non visqueux, donc quasi linéaire en alpha).

    Utile si l'on choisit d'optimiser à portance imposée (vol de croisière
    L = W) plutôt qu'à incidence imposée : alpha devient alors une SORTIE du
    modèle aéro, et une ENTRÉE du modèle SER (attitude du profil vue du radar).
    """
    X, Y = naca4(m, p, t, n_half=n_half)
    cache = {}
    f = lambda a: solve(X, Y, a, cache=cache).CL - CL_target
    return brentq(f, *bracket, xtol=1e-6)


def evaluate_at_cl(m, p, t, CL_target, Re=3e6, **kw):
    a = alpha_for_cl(m, p, t, CL_target, n_half=kw.get("n_half", 100))
    return evaluate(m, p, t, alpha=a, Re=Re, **kw)
