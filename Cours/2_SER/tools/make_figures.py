"""Figures pgfplots du cours SER, calculées par le code du projet (Code/ser, Code/aero).

Usage (depuis LaTeX/) : python3 tools/make_figures.py
Les chiffres cités dans le texte sont écrits dans figures/chiffres.json.
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJ = os.path.abspath(os.path.join(HERE, "..", ".."))   # racine du projet HackathonQuantique
CODE = os.path.join(PROJ, "Code")
sys.path.insert(0, CODE)

from aero.naca4 import naca4, leading_edge_radius  # noqa: E402
from ser.po2d import wavenumber, echo_width, radar_directions, to_db  # noqa: E402
from ser.model import evaluate_ser, RadarCase  # noqa: E402

FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
NUM = {}
K10 = wavenumber(10e9)


def write(name, body):
    with open(os.path.join(FIG, name + ".tex"), "w") as f:
        f.write(f"% figure générée par tools/make_figures.py : {name}\n{body}\n")


def co(xs, ys, fmt="{:.5g}"):
    return " ".join(f"({fmt.format(float(x))},{fmt.format(float(y))})" for x, y in zip(xs, ys)
                    if np.isfinite(x) and np.isfinite(y))


# ------------------------------------------------------------ validation : bande et cylindre
def fig_validation():
    w = 0.3
    e = 1e-7
    X = np.array([w / 2, -w / 2, -w / 2, w / 2, w / 2])
    Y = np.array([e, e, -e, -e, e])
    th = np.linspace(-20, 20, 801)
    t = np.deg2rad(th)
    s = np.column_stack([np.sin(t), np.cos(t)])
    num = echo_width(X, Y, s, K10)
    ana = K10 * w**2 * np.cos(t) ** 2 * np.sinc(K10 * w * np.sin(t) / np.pi) ** 2
    NUM["strip_maxerr_db"] = float(np.max(np.abs(to_db(num[ana > 1e-6]) - to_db(ana[ana > 1e-6]))))
    k = slice(None, None, 3)
    # cylindre
    ns = np.array([16, 32, 64, 128, 256, 512, 1024])
    errs = []
    for a in (0.05, 0.2):
        ee = []
        for n in ns:
            ang = np.linspace(0, 2 * np.pi, n + 1)
            sig = echo_width(a * np.cos(ang), a * np.sin(ang), np.array([[1.0, 0.0]]), K10)[0]
            ee.append(sig / (np.pi * a))
        errs.append(ee)
    NUM["cyl_ratio_fine"] = [float(errs[0][-1]), float(errs[1][-1])]
    write("validation", rf"""\begin{{tikzpicture}}
\begin{{groupplot}}[group style={{group size=2 by 1,horizontal sep=1.7cm}},width=0.48\linewidth,height=5.4cm,
  title style={{font=\small}},tick label style={{font=\scriptsize}},legend style={{draw=none,font=\scriptsize}}]
\nextgroupplot[title={{Bande de largeur $w=0{{,}}3$~m, 10~GHz}},xlabel={{angle depuis la normale (°)}},ylabel={{$\sigma_{{2D}}$ (dBm)}},ymin=-30,ymax=16,legend pos=north east]
\addplot[cblue,very thick] coordinates {{{co(th[k], to_db(num[k]))}}};\addlegendentry{{code PO}}
\addplot[corange,dashed,thick] coordinates {{{co(th[k], to_db(ana[k]))}}};\addlegendentry{{formule exacte}}
\nextgroupplot[title={{Cylindre : $\sigma_{{2D}}/(\pi a)$}},xlabel={{nombre de facettes}},xmode=log,ymin=0.5,ymax=1.1,legend pos=south east,
  xtick={{16,64,256,1024}},xticklabels={{16,64,256,1024}}]
\addplot[cblue,very thick,mark=*,mark size=1.3pt] coordinates {{{co(ns, errs[0])}}};\addlegendentry{{$a=5$~cm ($ka=10$)}}
\addplot[corange,very thick,mark=square*,mark size=1.3pt] coordinates {{{co(ns, errs[1])}}};\addlegendentry{{$a=20$~cm ($ka=42$)}}
\addplot[cmuted,dashed] coordinates {{(16,1) (1024,1)}};
\end{{groupplot}}
\end{{tikzpicture}}""")


# ------------------------------------------------------------ diagrammes de profils
def fig_diagrammes():
    eps = np.linspace(-90, 90, 721)
    pl = []
    for code, st in (("0012", "cmuted"), ("2412", "cblue"), ("4415", "corange")):
        m, p, t = int(code[0]) / 100, int(code[1]) / 10, int(code[2:]) / 100
        X, Y = naca4(m, p, t, n_half=200)
        sig = echo_width(X, Y, radar_directions(0.0, eps), K10)
        pl.append(f"\\addplot[{st},thick] coordinates {{{co(eps[::2], to_db(sig[::2]))}}};\\addlegendentry{{NACA {code}}}")
        NUM[f"nose_{code}"] = float(to_db(sig[360]))
        NUM[f"pi_rle_{code}"] = float(to_db(np.pi * leading_edge_radius(t)))
    write("diagrammes", rf"""\begin{{tikzpicture}}
\begin{{axis}}[width=0.9\linewidth,height=6.2cm,xlabel={{élévation du radar $\varepsilon$ (°), $\alpha=0$ ; $\varepsilon>0$ : radar au-dessus}},
  ylabel={{$\sigma_{{2D}}$ (dBm)}},xmin=-90,xmax=90,ymin=-25,ymax=15,xtick={{-90,-60,...,90}},
  legend style={{draw=none,font=\scriptsize,at={{(0.5,0.98)}},anchor=north}},legend columns=3]
\fill[cgreen!10] (axis cs:-20,-25) rectangle (axis cs:20,15);
\node[cgreen!60!black,font=\scriptsize,anchor=south] at (axis cs:0,-25) {{secteur S1}};
{chr(10).join(pl)}
\end{{axis}}
\end{{tikzpicture}}""")


# ------------------------------------------------------------ SER de nez contre rayon de BA
def fig_rayon():
    ts = np.linspace(0.06, 0.20, 15)
    nose, mean = [], []
    for t in ts:
        X, Y = naca4(0.0, 0.0, t, n_half=200)
        nose.append(echo_width(X, Y, radar_directions(0.0, [0.0]), K10)[0])
    th = np.linspace(0.05, 0.21, 50)
    write("rayon", rf"""\begin{{tikzpicture}}
\begin{{axis}}[width=0.7\linewidth,height=5.6cm,xlabel={{épaisseur relative $t$}},ylabel={{$\sigma_{{2D}}$ de nez (dBm)}},
  legend style={{draw=none,font=\scriptsize}},legend pos=south east,
  xticklabel style={{/pgf/number format/fixed,/pgf/number format/use comma}}]
\addplot[cblue,only marks,mark=*,mark size=1.8pt] coordinates {{{co(ts, to_db(np.array(nose)))}}};\addlegendentry{{PO, profils symétriques}}
\addplot[corange,thick,domain=0.05:0.21,samples=50] {{10*log10(pi*1.1019*x^2)}};\addlegendentry{{$\pi\,r_{{BA}}=1{{,}}1019\,\pi t^2$}}
\end{{axis}}
\end{{tikzpicture}}""")
    NUM["nose_t006"] = float(to_db(nose[0]))
    NUM["nose_t020"] = float(to_db(nose[-1]))


# ------------------------------------------------------------ facettage et fréquence
def fig_convergence():
    ns = [25, 50, 100, 150, 200, 300, 400]
    out = {}
    for code in ("0012", "2412"):
        m, p, t = int(code[0]) / 100, int(code[1]) / 10, int(code[2:]) / 100
        out[code] = [evaluate_ser(m, p, t, 2.0, RadarCase(n_half=n)).ser_db for n in ns]
    NUM["conv_2412"] = dict(zip(ns, out["2412"]))
    fr = np.geomspace(1e9, 40e9, 30)
    fsig = [evaluate_ser(0.02, 0.4, 0.12, 2.0, RadarCase(freq_hz=f)).ser_db for f in fr]
    NUM["freq_2412"] = dict(f1=fsig[0], f10=float(np.interp(10e9, fr, fsig)), f40=fsig[-1])
    write("convergence", rf"""\begin{{tikzpicture}}
\begin{{groupplot}}[group style={{group size=2 by 1,horizontal sep=1.7cm}},width=0.48\linewidth,height=5.4cm,
  title style={{font=\small}},tick label style={{font=\scriptsize}},legend style={{draw=none,font=\scriptsize}}]
\nextgroupplot[title={{Facettage (secteur S1, 10~GHz)}},xlabel={{\texttt{{n\_half}}}},ylabel={{SER moyenne (dBm)}},legend pos=south east]
\addplot[cmuted,very thick,mark=*,mark size=1.3pt] coordinates {{{co(ns, out['0012'])}}};\addlegendentry{{NACA 0012}}
\addplot[cblue,very thick,mark=square*,mark size=1.3pt] coordinates {{{co(ns, out['2412'])}}};\addlegendentry{{NACA 2412}}
\nextgroupplot[title={{Fréquence (NACA 2412, $\alpha=2^\circ$)}},xlabel={{fréquence (GHz)}},xmode=log,
  xtick={{1,2,5,10,20,40}},xticklabels={{1,2,5,10,20,40}}]
\addplot[corange,very thick,mark=*,mark size=1.1pt] coordinates {{{co(fr / 1e9, fsig)}}};
\end{{groupplot}}
\end{{tikzpicture}}""")


# ------------------------------------------------------------ point spéculaire et courbure
def fig_speculaire():
    """sigma(eps) PO comparé à pi * rho(point spéculaire), rho mesuré sur la géométrie."""
    m, p, t = 0.0, 0.0, 0.12
    X, Y = naca4(m, p, t, n_half=400)
    eps = np.linspace(-60, 60, 241)
    sig = echo_width(X, Y, radar_directions(0.0, eps), K10)
    # rayon de courbure local par cercle passant par 3 points
    xc, yc = 0.5 * (X[:-1] + X[1:]), 0.5 * (Y[:-1] + Y[1:])
    tx, ty = np.diff(X), np.diff(Y)
    L = np.hypot(tx, ty)
    nx, ny = ty / L, -tx / L
    ang_n = np.degrees(np.arctan2(ny, nx))
    dang = np.gradient(np.unwrap(np.arctan2(ny, nx)))
    rho = L / np.maximum(np.abs(dang), 1e-12)
    pred = []
    for e in eps:
        s = radar_directions(0.0, [e])[0]
        target = np.degrees(np.arctan2(s[1], s[0]))
        j = np.argmin(np.abs((ang_n - target + 180) % 360 - 180))
        pred.append(np.pi * rho[j])
    write("speculaire", rf"""\begin{{tikzpicture}}
\begin{{axis}}[width=0.75\linewidth,height=5.6cm,xlabel={{élévation $\varepsilon$ (°)}},ylabel={{$\sigma_{{2D}}$ (dBm)}},
  xmin=-60,xmax=60,legend style={{draw=none,font=\scriptsize}},legend pos=north west]
\addplot[cblue,very thick] coordinates {{{co(eps, to_db(sig))}}};\addlegendentry{{PO (NACA 0012)}}
\addplot[corange,dashed,very thick] coordinates {{{co(eps, to_db(np.array(pred)))}}};\addlegendentry{{$\pi\rho$ au point spéculaire}}
\end{{axis}}
\end{{tikzpicture}}""")


# ------------------------------------------------------------ cartes sur la grille
def fig_cartes():
    CODEDIR = os.path.join(CODE, "resultats")
    blocks = []
    for name, title in (("S1_note", "S1 : face, $\\varepsilon\\in[-20^\\circ,20^\\circ]$"),
                        ("S2_sol", "S2 : radar en contrebas, $\\varepsilon\\in[-60^\\circ,0^\\circ]$")):
        rows = list(csv.DictReader(open(os.path.join(CODEDIR, f"base_complete_{name}.csv"))))
        g = sorted([(int(r["im"]), float(r["t"]) * 100, float(r["SER_db"])) for r in rows if int(r["ip"]) == 4],
                   key=lambda z: (z[1], z[0]))
        pts = " ".join(f"({im},{t:.0f}) [{v:.3f}]" for im, t, v in g)
        blocks.append(rf"""\nextgroupplot[title={{{title}}},colorbar,colorbar style={{ylabel={{SER (dBm)}},width=2.5mm,tick label style={{font=\scriptsize}}}}]
\addplot[matrix plot*,mesh/cols=8,point meta=explicit] coordinates {{{pts}}};""")
        S = np.array([float(r["SER_db"]) for r in rows])
        t = np.array([float(r["t"]) for r in rows])
        m = np.array([float(r["m"]) for r in rows])
        NUM[name] = dict(min=float(S.min()), max=float(S.max()), corr_t=float(np.corrcoef(S, t)[0, 1]),
                         corr_m=float(np.corrcoef(S, m)[0, 1]))
    write("cartes", rf"""\begin{{tikzpicture}}
\begin{{groupplot}}[group style={{group size=2 by 1,horizontal sep=3.2cm}},width=0.36\linewidth,height=5.6cm,
  xlabel={{cambrure $m$ (\%)}},ylabel={{épaisseur $t$ (\%)}},xtick={{0,...,7}},ytick={{6,8,...,20}},
  enlargelimits=false,axis on top,title style={{font=\small}},tick label style={{font=\scriptsize}},
  colormap={{bl}}{{color=(white) color=(cblue!45) color=(cblue!85!black)}}]
{chr(10).join(blocks)}
\end{{groupplot}}
\end{{tikzpicture}}""")


# ------------------------------------------------------------ validité de la PO : cylindre exact (série de Mie)
def fig_mie():
    from scipy.special import jv, hankel2, jvp, h2vp

    def exact(ka, pol):
        N = int(ka + 10 * ka ** (1 / 3) + 15)
        n = np.arange(-N, N + 1)
        sg = (-1.0) ** n
        if pol == "TM":
            sm = np.sum(sg * jv(n, ka) / hankel2(n, ka))
        else:
            sm = np.sum(sg * jvp(n, ka) / h2vp(n, ka))
        return 4 * abs(sm) ** 2 / ka / np.pi          # sigma / (pi a)
    ka = np.geomspace(0.3, 60, 70)
    tm = [10 * np.log10(exact(x, "TM")) for x in ka]
    te = [10 * np.log10(exact(x, "TE")) for x in ka]
    NUM["mie"] = {f"{x}": [float(10 * np.log10(exact(x, "TM"))), float(10 * np.log10(exact(x, "TE")))]
                  for x in (0.75, 1.0, 3.3, 10.0)}
    r06, r12, r20 = (1.1019 * t**2 * K10 for t in (0.06, 0.12, 0.20))
    NUM["kr_le"] = dict(t06=r06, t12=r12, t20=r20)
    marks = "\n".join(
        rf"\draw[cgreen!60!black,thick] (axis cs:{x:.3f},-3) -- (axis cs:{x:.3f},-1.2) node[pos=1,anchor=west,rotate=90,font=\scriptsize] {{$t={lab}$\,\%}};"
        for x, lab in ((r06, 6), (r12, 12), (r20, 20)))
    write("mie", rf"""\begin{{tikzpicture}}
\begin{{axis}}[width=0.75\linewidth,height=5.6cm,xmode=log,xlabel={{$ka$}},ylabel={{$10\log_{{10}}\big(\sigma_{{\text{{exact}}}}/\pi a\big)$ (dB)}},
  xmin=0.3,xmax=60,ymin=-3,ymax=3,legend style={{draw=none,font=\scriptsize,at={{(0.5,1.03)}},anchor=south}},legend columns=2,
  xtick={{0.5,1,2,5,10,20,50}},xticklabels={{0{{,}}5,1,2,5,10,20,50}}]
\addplot[cblue,very thick] coordinates {{{co(ka, tm)}}};\addlegendentry{{polarisation TM ($E$ parallèle à l'envergure)}}
\addplot[corange,very thick] coordinates {{{co(ka, te)}}};\addlegendentry{{polarisation TE}}
\addplot[cmuted,dashed] coordinates {{(0.3,0) (60,0)}};
{marks}
\end{{axis}}
\end{{tikzpicture}}""")


if __name__ == "__main__":
    for f in (fig_validation, fig_diagrammes, fig_rayon, fig_convergence, fig_speculaire, fig_cartes, fig_mie):
        f()
        print("ok", f.__name__)
    with open(os.path.join(FIG, "chiffres.json"), "w") as fh:
        json.dump(NUM, fh, indent=1, default=float)
    print(json.dumps(NUM, indent=1, default=float))
