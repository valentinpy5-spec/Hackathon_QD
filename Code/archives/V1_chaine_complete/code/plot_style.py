"""Style commun des figures (palette catégorielle validée, axes discrets)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_UP = "#2a78d6"      # extrados (série 1)
C_LO = "#eb6834"      # intrados (série 2)
C_REF = "#1baf7a"     # référence / théorie (série 3)
INK = "#52514e"
SEQ_CMAP = "Blues"    # magnitude : une seule teinte, clair -> foncé

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": INK,
    "axes.labelcolor": "#0b0b0b",
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.grid": True,
    "grid.color": "#e4e3df",
    "grid.linewidth": 0.6,
    "lines.linewidth": 2.0,
    "legend.frameon": False,
    "font.size": 10,
})
