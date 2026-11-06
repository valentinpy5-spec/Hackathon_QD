"""Génère les figures pgfplots « à données » du cours sur le modèle aéro (phase 1).

Toutes les courbes sont calculées en appelant le code du projet
(Code/aero) ; les coordonnées sont embarquées
dans figures/*.tex, si bien que le .tex se compile sans Python.
Les nombres cités dans le texte sont écrits dans figures/chiffres.json.

Usage (depuis LaTeX/) : python3 tools/make_figures.py
"""
import csv
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJ = os.path.abspath(os.path.join(HERE, "..", ".."))   # racine du projet HackathonQuantique
CODE = os.path.join(PROJ, "Code")
sys.path.insert(0, CODE)

from aero import naca4, naca4_from_code, solve, evaluate, alpha_for_cl  # noqa: E402
from aero.naca4 import camber_line  # noqa: E402
from aero.boundary_layer import (march_surface, split_at_stagnation, thwaites_closure,  # noqa: E402
                                 michel_transition, H1_of_H, cf_ludwieg_tillmann)

FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
NUM = {}


def write(name, body):
    with open(os.path.join(FIG, name + ".tex"), "w") as f:
        f.write(f"% figure générée par tools/make_figures.py : {name}\n{body}\n")


def co(xs, ys, fmt="{:.5g}"):
    return " ".join(f"({fmt.format(float(x))},{fmt.format(float(y))})" for x, y in zip(xs, ys)
                    if np.isfinite(x) and np.isfinite(y))


def up_lo(X, Y):
    n = (len(X) - 1) // 2
    return (X[: n + 1][::-1], Y[: n + 1][::-1]), (X[n:], Y[n:])


# ===================================================================== géométrie
def fig_profils():
    plots = []
    for code, style in (("0012", "cmuted,thick"), ("2412", "cblue,very thick"), ("4415", "corange,very thick")):
        X, Y = naca4_from_code(code, n_half=80)
        plots.append(f"\\addplot[{style}] coordinates {{{co(X, Y)}}};\n\\addlegendentry{{NACA {code}}}")
    write("profils", rf"""\begin{{tikzpicture}}
\begin{{axis}}[width=\linewidth,height=4.6cm,axis equal image,xmin=-0.02,xmax=1.02,ymin=-0.12,ymax=0.16,
  xlabel={{$x/c$}},ylabel={{$y/c$}},legend columns=3,legend style={{draw=none,at={{(0.5,1.03)}},anchor=south,font=\small}}]
{chr(10).join(plots)}
\end{{axis}}
\end{{tikzpicture}}""")


def fig_parametres():
    groups = [("m", [(0.0, 0.4, 0.12), (0.03, 0.4, 0.12), (0.06, 0.4, 0.12)], ["0", "3~\\%", "6~\\%"],
               "effet de $m$ ($p=0{,}4$, $t=12$~\\%)"),
              ("p", [(0.05, 0.2, 0.12), (0.05, 0.4, 0.12), (0.05, 0.55, 0.12)], ["0{,}20", "0{,}40", "0{,}55"],
               "effet de $p$ ($m=5$~\\%, $t=12$~\\%)"),
              ("t", [(0.02, 0.4, 0.06), (0.02, 0.4, 0.12), (0.02, 0.4, 0.20)], ["6~\\%", "12~\\%", "20~\\%"],
               "effet de $t$ ($m=2$~\\%, $p=0{,}4$)")]
    styles = ["cmuted,thick", "cblue,very thick", "corange,very thick"]
    out = []
    for name, cases, labs, title in groups:
        pl = []
        for (m, p, t), lab, st in zip(cases, labs, styles):
            X, Y = naca4(m, p, t, n_half=60)
            pl.append(f"\\addplot[{st}] coordinates {{{co(X, Y)}}};\\addlegendentry{{${name}={lab}$}}")
            if name != "t":
                xc = np.linspace(0, 1, 60)
                yc, _ = camber_line(xc, m, p)
                pl.append(f"\\addplot[{st.split(',')[0]},dashed,thin,forget plot] coordinates {{{co(xc, yc)}}};")
        out.append(f"\\nextgroupplot[title={{{title}}}]\n" + "\n".join(pl))
    write("parametres", rf"""\begin{{tikzpicture}}
\begin{{groupplot}}[group style={{group size=1 by 3,vertical sep=1.5cm,x descriptions at=edge bottom}},width=\linewidth,height=3.9cm,
  axis equal image,xmin=-0.02,xmax=1.02,ymin=-0.12,ymax=0.17,xlabel={{$x/c$}},ylabel={{$y/c$}},
  title style={{font=\small}},legend columns=3,legend style={{draw=none,font=\scriptsize,at={{(0.99,0.97)}},anchor=north east,fill=white,fill opacity=0.8,text opacity=1}}]
{chr(10).join(out)}
\end{{groupplot}}
\end{{tikzpicture}}""")


def fig_cosinus():
    n = 30
    X, Y = naca4(0.0, 0.0, 0.12, n_half=n)
    xu = np.linspace(0, 1, n + 1)
    from aero.naca4 import thickness
    yu = thickness(xu, 0.12)
    up = X[: n + 1][::-1], Y[: n + 1][::-1]
    xx = np.linspace(0, 1, 400)
    write("cosinus", rf"""\begin{{tikzpicture}}
\begin{{axis}}[width=0.62\linewidth,height=5cm,xmin=-0.005,xmax=0.2,ymin=0,ymax=0.065,scaled ticks=false,
  xtick={{0,0.05,0.1,0.15,0.2}},ytick={{0,0.02,0.04,0.06}},tick label style={{/pgf/number format/fixed,/pgf/number format/use comma,font=\small}},
  xlabel={{$x/c$}},ylabel={{$y/c$}},legend style={{draw=none,font=\small,at={{(0.98,0.04)}},anchor=south east}}]
\addplot[cmuted,thin,forget plot] coordinates {{{co(xx, thickness(xx, 0.12))}}};
\addplot[only marks,mark=*,mark size=1.6pt,cblue] coordinates {{{co(*up)}}};
\addlegendentry{{cosinus ({2*n} panneaux)}}
\addplot[only marks,mark=square*,mark size=1.6pt,corange] coordinates {{{co(xu, yu)}}};
\addlegendentry{{uniforme ({2*n} panneaux)}}
\end{{axis}}
\end{{tikzpicture}}""")
    NUM["cos_n_le"] = int(np.sum(up[0] < 0.05))
    NUM["uni_n_le"] = int(np.sum(xu < 0.05))


# ===================================================================== panneaux
def cp_curves(code, alpha, n_half=100):
    X, Y = naca4_from_code(code, n_half=n_half)
    s = solve(X, Y, alpha)
    P = s.panels
    n = n_half
    return (P.xc[:n][::-1], s.Cp[:n][::-1]), (P.xc[n:], s.Cp[n:]), s


def fig_cp():
    blocks = []
    for code, a, title in (("0012", 0.0, "NACA 0012, $\\alpha=0^\\circ$"),
                           ("0012", 5.0, "NACA 0012, $\\alpha=5^\\circ$"),
                           ("2412", 2.0, "NACA 2412, $\\alpha=2^\\circ$")):
        (xu, cu), (xl, cl), s = cp_curves(code, a)
        blocks.append(rf"""\nextgroupplot[title={{{title}}}]
\addplot[cblue,very thick] coordinates {{{co(xu, cu)}}};
\addplot[corange,very thick] coordinates {{{co(xl, cl)}}};
\node[font=\scriptsize,anchor=south east,fill=white,inner sep=1pt] at (rel axis cs:0.98,0.02) {{$C_L={s.CL:.3f}$}};""")
        NUM[f"CL_{code}_a{a:g}"] = s.CL
        NUM[f"Cpmin_{code}_a{a:g}"] = float(min(cu.min(), cl.min()))
    write("cp", rf"""\begin{{tikzpicture}}
\begin{{groupplot}}[group style={{group size=3 by 1,horizontal sep=0.9cm,y descriptions at=edge left}},
  width=0.34\linewidth,height=5.2cm,y dir=reverse,xmin=0,xmax=1,ymin=-2.6,ymax=1.1,
  xlabel={{$x/c$}},ylabel={{$C_p$}},title style={{font=\small}},tick label style={{font=\scriptsize}}]
{chr(10).join(blocks)}
\end{{groupplot}}
\end{{tikzpicture}}""")


def fig_convergence():
    Ns = np.array([10, 15, 20, 30, 40, 60, 80, 100, 160, 240, 320])
    X, Y = naca4_from_code("0012", n_half=1200)
    ref = solve(X, Y, 5.0).CL
    eCL, eKJ, cdp = [], [], []
    for n in Ns:
        X, Y = naca4_from_code("0012", n_half=int(n))
        s = solve(X, Y, 5.0)
        eCL.append(abs(s.CL - ref) / ref)
        eKJ.append(abs(s.CL - s.CL_kj) / ref)
        cdp.append(abs(s.CD_p))
    N = 2 * Ns
    slope = np.polyfit(np.log(N[3:]), np.log(np.array(eCL)[3:]), 1)[0]
    slope_d = np.polyfit(np.log(N[3:]), np.log(np.array(cdp)[3:]), 1)[0]
    NUM.update(conv_slope_CL=float(slope), conv_slope_CDp=float(slope_d), CL_ref_0012_a5=float(ref),
               conv_err_200=float(eCL[list(Ns).index(100)]))
    write("convergence", rf"""\begin{{tikzpicture}}
\begin{{loglogaxis}}[width=0.72\linewidth,height=6cm,xlabel={{nombre de panneaux $N$}},ylabel={{erreur}},
  legend style={{draw=none,font=\small,at={{(0.02,0.02)}},anchor=south west}},xmin=15,xmax=800,xtick={{20,50,100,200,500}},xticklabels={{20,50,100,200,500}}]
\addplot[cblue,very thick,mark=*,mark size=1.5pt] coordinates {{{co(N, eCL, "{:.4g}")}}};
\addlegendentry{{$|C_L(N)-C_L^\infty|/C_L^\infty$}}
\addplot[corange,very thick,mark=square*,mark size=1.5pt] coordinates {{{co(N, eKJ, "{:.4g}")}}};
\addlegendentry{{$|C_{{L,C_p}}-C_{{L,KJ}}|/C_L^\infty$}}
\addplot[cgreen,very thick,mark=triangle*,mark size=1.8pt] coordinates {{{co(N, cdp, "{:.4g}")}}};
\addlegendentry{{$|C_{{D,p}}|$ (d'Alembert)}}
\addplot[cmuted,dashed,domain=40:640] {{0.6/x}};
\node[cmuted,font=\scriptsize] at (axis cs:330,0.0035) {{pente $-1$}};
\end{{loglogaxis}}
\end{{tikzpicture}}""")


def alpha_l0_thin(m, p):
    th = np.linspace(0, np.pi, 4001)
    x = 0.5 * (1 - np.cos(th))
    _, dyc = camber_line(x, m, p)
    return -np.trapezoid(dyc * (np.cos(th) - 1), th) / np.pi


def fig_cl_alpha():
    a = np.arange(-4, 10.5, 1.0)
    pl = []
    for code, st in (("0012", "cmuted"), ("2412", "cblue"), ("4412", "corange")):
        X, Y = naca4_from_code(code, n_half=100)
        cache = {}
        cl = [solve(X, Y, ai, cache=cache).CL for ai in a]
        m, p = int(code[0]) / 100, int(code[1]) / 10
        a0 = np.rad2deg(alpha_l0_thin(m, p))
        pl.append(f"\\addplot[{st},very thick,mark=*,mark size=1.3pt] coordinates {{{co(a, cl)}}};\\addlegendentry{{NACA {code} (panneaux)}}")
        pl.append(f"\\addplot[{st},dashed,domain=-4:10] {{2*pi*(x-({a0:.4f}))*pi/180}};\\addlegendentry{{profil mince, $\\alpha_{{L0}}={a0:.2f}^\\circ$}}")
        sl = np.polyfit(np.deg2rad(a), cl, 1)
        NUM[f"slope_{code}"] = float(sl[0])
        NUM[f"aL0_panel_{code}"] = float(np.rad2deg(-sl[1] / sl[0]))
        NUM[f"aL0_thin_{code}"] = float(a0)
    write("cl_alpha", rf"""\begin{{tikzpicture}}
\begin{{axis}}[width=0.75\linewidth,height=6.4cm,xlabel={{$\alpha$ (°)}},ylabel={{$C_L$ (non visqueux)}},
  xmin=-4,xmax=10,legend columns=2,legend style={{draw=none,font=\scriptsize,at={{(0.5,-0.22)}},anchor=north}},legend cell align=left]
{chr(10).join(pl)}
\end{{axis}}
\end{{tikzpicture}}""")


def fig_vt():
    X, Y = naca4_from_code("2412", n_half=100)
    s = solve(X, Y, 2.0)
    P = s.panels
    sn = np.concatenate([[0], np.cumsum(P.L)])
    sc = 0.5 * (sn[:-1] + sn[1:])
    up, lo = split_at_stagnation(s)
    k = up[3][0]
    w = s.Vt[k] / (s.Vt[k] - s.Vt[k + 1])
    s_st = sc[k] + w * (sc[k + 1] - sc[k])
    x_st = P.xc[k] + w * (P.xc[k + 1] - P.xc[k])
    NUM["x_stag_2412_a2"] = float(x_st)
    write("vt_contour", rf"""\begin{{tikzpicture}}
\begin{{groupplot}}[group style={{group size=2 by 1,horizontal sep=1.6cm}},width=0.5\linewidth,height=5.4cm,
  title style={{font=\small}},tick label style={{font=\scriptsize}},legend style={{draw=none,font=\scriptsize}}]
\nextgroupplot[title={{$V_t$ signée le long du contour}},xlabel={{abscisse $\ell$ depuis le BF extrados}},ylabel={{$V_t/U_\infty$}},xmin=0,xmax={sn[-1]:.3f}]
\addplot[cink,very thick] coordinates {{{co(sc, s.Vt)}}};
\addplot[cmuted,thin] coordinates {{(0,0) ({sn[-1]:.3f},0)}};
\addplot[only marks,mark=*,corange,mark size=2.4pt] coordinates {{({s_st:.4f},0)}};
\node[corange,font=\scriptsize,anchor=north west] at (axis cs:{s_st+0.03:.3f},-0.08) {{point d'arrêt}};
\node[font=\scriptsize,anchor=north] at (axis cs:0.5,-0.2) {{extrados ($V_t<0$)}};
\node[font=\scriptsize,anchor=north] at (axis cs:1.6,0.8) {{intrados ($V_t>0$)}};
\nextgroupplot[title={{$U_e=|V_t|$ sur chaque face}},xlabel={{$x/c$}},ylabel={{$U_e/U_\infty$}},xmin=0,xmax=1,legend pos=south east]
\addplot[cblue,very thick] coordinates {{{co(up[1], up[2])}}};\addlegendentry{{extrados}}
\addplot[corange,very thick] coordinates {{{co(lo[1], lo[2])}}};\addlegendentry{{intrados}}
\end{{groupplot}}
\end{{tikzpicture}}""")


# ===================================================================== couche limite
def fig_thwaites_closure():
    lam = np.linspace(-0.09, 0.1, 120)
    H, ell = thwaites_closure(lam)
    F = 2 * (ell - (2 + H) * lam)
    write("thwaites_fermeture", rf"""\begin{{tikzpicture}}
\begin{{groupplot}}[group style={{group size=2 by 1,horizontal sep=1.6cm}},width=0.5\linewidth,height=5.4cm,
  xlabel={{$\lambda=\theta^2U_e'/\nu$}},xmin=-0.1,xmax=0.1,title style={{font=\small}},tick label style={{font=\scriptsize}},
  legend style={{draw=none,font=\scriptsize}}]
\nextgroupplot[title={{Corrélations de fermeture}},legend pos=north east,ymin=0,ymax=4]
\addplot[cblue,very thick] coordinates {{{co(lam, H)}}};\addlegendentry{{$H(\lambda)$}}
\addplot[corange,very thick] coordinates {{{co(lam, 10 * ell)}}};\addlegendentry{{$10\,\ell(\lambda)$}}
\addplot[cmuted,dashed] coordinates {{(-0.09,0) (-0.09,4)}};
\node[cmuted,font=\scriptsize,rotate=90,anchor=south] at (axis cs:-0.09,2.6) {{décollement}};
\nextgroupplot[title={{$F(\lambda)=2[\ell-(2+H)\lambda]$ et sa droite}},legend pos=north east]
\addplot[cblue,very thick] coordinates {{{co(lam, F)}}};\addlegendentry{{$F$ exact (corrélations)}}
\addplot[corange,dashed,very thick,domain=-0.09:0.1] {{0.45-6*x}};\addlegendentry{{$0{{,}}45-6\lambda$}}
\end{{groupplot}}
\end{{tikzpicture}}""")


def fig_plaque():
    Re = 1e7
    s = np.linspace(1e-4, 1.0, 400)
    Ue = np.ones_like(s)
    lam = march_surface(s, s, Ue, Re, free_transition=False, x_stop=2.0)
    tur = march_surface(s, s, Ue, Re, x_trip=0.0, free_transition=False, x_stop=2.0)
    Rex = Re * s
    thb = 0.664 * s / np.sqrt(Rex)
    th7 = 0.036 * s * Rex ** -0.2
    k = slice(None, None, 8)
    NUM["plaque_err_lam"] = float(np.max(np.abs(lam.theta[50:] / thb[50:] - 1)))
    i = s > 0.3
    NUM["plaque_err_turb"] = float(np.max(np.abs(tur.theta[i] / th7[i] - 1)))
    NUM["plaque_H_lam"] = float(lam.H[-1])
    NUM["plaque_H_turb"] = float(tur.H[-1])
    write("plaque_plane", rf"""\begin{{tikzpicture}}
\begin{{groupplot}}[group style={{group size=2 by 1,horizontal sep=1.6cm}},width=0.5\linewidth,height=5.2cm,
  xlabel={{$x/c$}},xmin=0,xmax=1,title style={{font=\small}},tick label style={{font=\scriptsize}},
  legend style={{draw=none,font=\scriptsize}},legend pos=north west]
\nextgroupplot[title={{Laminaire (Thwaites), $Re=10^7$}},ylabel={{$10^3\,\theta/c$}}]
\addplot[cblue,very thick] coordinates {{{co(s[k], 1e3 * lam.theta[k])}}};\addlegendentry{{Thwaites (code)}}
\addplot[cgreen,dashed,very thick] coordinates {{{co(s[k], 1e3 * thb[k])}}};\addlegendentry{{Blasius}}
\nextgroupplot[title={{Turbulent dès $x=0$ (Head)}}]
\addplot[corange,very thick] coordinates {{{co(s[k], 1e3 * tur.theta[k])}}};\addlegendentry{{Head (code)}}
\addplot[cgreen,dashed,very thick] coordinates {{{co(s[k], 1e3 * th7[k])}}};\addlegendentry{{loi en $1/7$}}
\end{{groupplot}}
\end{{tikzpicture}}""")


def thwaites_raw(s, Ue, Re):
    s0 = np.concatenate([[0.0], s])
    U0 = np.concatenate([[0.0], Ue])
    integ = np.concatenate([[0.0], np.cumsum(0.5 * (U0[1:]**5 + U0[:-1]**5) * np.diff(s0))])[1:]
    th = np.sqrt(0.45 / Re * integ / Ue**6)
    lam = th**2 * np.gradient(Ue, s) * Re
    return th, lam


def fig_transition():
    Re = 3e6
    r, inv, (bu, bl) = evaluate(0.02, 0.4, 0.12, alpha=2.0, Re=Re, return_details=True)
    up, lo = split_at_stagnation(inv)
    blocks = []
    for (s, x, Ue, _), b, title in ((up, bu, "extrados"), (lo, bl, "intrados")):
        keep = x <= 0.985
        s, x, Ue = s[keep], x[keep], Ue[keep]
        th, lam = thwaites_raw(s, Ue, Re)
        Ret = Ue * th * Re
        Rex = np.maximum(Ue * s * Re, 1)
        mi = michel_transition(Rex)
        j = x > 0.003
        blocks.append(rf"""\nextgroupplot[title={{{title} : transition en $x/c={b.x_tr:.3f}$ ({b.tr_mode})}}]
\addplot[cblue,very thick] coordinates {{{co(x[j], Ret[j])}}};\addlegendentry{{$Re_\theta$ (Thwaites)}}
\addplot[corange,very thick] coordinates {{{co(x[j], mi[j])}}};\addlegendentry{{seuil de Michel}}
\addplot[cmuted,dashed] coordinates {{({b.x_tr:.4f},0) ({b.x_tr:.4f},2000)}};""")
    NUM.update(xtr_up_2412=bu.x_tr, xtr_lo_2412=bl.x_tr)
    write("transition_michel", rf"""\begin{{tikzpicture}}
\begin{{groupplot}}[group style={{group size=2 by 1,horizontal sep=1.6cm}},width=0.5\linewidth,height=5.4cm,
  xlabel={{$x/c$}},xmin=0,xmax=1,ymin=0,ymax=2000,title style={{font=\small}},tick label style={{font=\scriptsize}},
  legend style={{draw=none,font=\scriptsize}},legend pos=north west]
{chr(10).join(blocks)}
\end{{groupplot}}
\end{{tikzpicture}}""")


def fig_bl_profiles():
    r, inv, (bu, bl) = evaluate(0.02, 0.4, 0.12, alpha=2.0, Re=3e6, return_details=True)
    j = lambda b: b.x > 0.003
    write("cl_2412", rf"""\begin{{tikzpicture}}
\begin{{groupplot}}[group style={{group size=3 by 1,horizontal sep=1.2cm}},width=0.35\linewidth,height=5cm,
  xlabel={{$x/c$}},xmin=0,xmax=1,title style={{font=\small}},tick label style={{font=\scriptsize}},
  legend style={{draw=none,font=\scriptsize}}]
\nextgroupplot[title={{$10^3\,\theta/c$}},legend pos=north west]
\addplot[cblue,very thick] coordinates {{{co(bu.x[j(bu)], 1e3*bu.theta[j(bu)])}}};\addlegendentry{{extrados}}
\addplot[corange,very thick] coordinates {{{co(bl.x[j(bl)], 1e3*bl.theta[j(bl)])}}};\addlegendentry{{intrados}}
\nextgroupplot[title={{facteur de forme $H$}},ymin=1,ymax=3]
\addplot[cblue,very thick] coordinates {{{co(bu.x[j(bu)], bu.H[j(bu)])}}};
\addplot[corange,very thick] coordinates {{{co(bl.x[j(bl)], bl.H[j(bl)])}}};
\nextgroupplot[title={{$10^3\,c_f$}},ymin=0,ymax=8]
\addplot[cblue,very thick] coordinates {{{co(bu.x[j(bu)], 1e3*bu.cf[j(bu)])}}};
\addplot[corange,very thick] coordinates {{{co(bl.x[j(bl)], 1e3*bl.cf[j(bl)])}}};
\end{{groupplot}}
\end{{tikzpicture}}""")
    NUM.update(CD_2412=r.CD, CDf_2412=r.CD_f, CL_2412=r.CL, CM_2412=r.CM,
               H_te_up=float(bu.H_te), H_te_lo=float(bl.H_te),
               th_te_up=float(bu.theta_te), th_te_lo=float(bl.theta_te),
               Ue_te_up=float(bu.Ue_te), Ue_te_lo=float(bl.Ue_te))


def fig_head_closure():
    H = np.linspace(1.15, 2.8, 120)
    H1 = H1_of_H(H)
    H1b = np.linspace(3.35, 12, 120)
    F = 0.0306 * (H1b - 3.0) ** -0.6169
    write("head_fermeture", rf"""\begin{{tikzpicture}}
\begin{{groupplot}}[group style={{group size=2 by 1,horizontal sep=1.6cm}},width=0.5\linewidth,height=5.2cm,
  title style={{font=\small}},tick label style={{font=\scriptsize}}]
\nextgroupplot[title={{$H_1(H)$}},xlabel={{$H$}},ylabel={{$H_1=(\delta-\delta^*)/\theta$}},ymin=3,ymax=12]
\addplot[cblue,very thick] coordinates {{{co(H, H1)}}};
\addplot[cmuted,dashed] coordinates {{(2.4,3) (2.4,12)}};
\node[cmuted,font=\scriptsize,anchor=south west] at (axis cs:2.42,8) {{$H_{{sep}}=2{{,}}4$}};
\nextgroupplot[title={{entraînement $F(H_1)$}},xlabel={{$H_1$}},ylabel={{$F=\frac{{1}}{{U_e}}\frac{{d(U_e\theta H_1)}}{{ds}}$}}]
\addplot[corange,very thick] coordinates {{{co(H1b, F)}}};
\end{{groupplot}}
\end{{tikzpicture}}""")


def fig_xtr_alpha():
    a = np.arange(-4, 10.5, 1.0)
    rs = [evaluate(0.02, 0.4, 0.12, alpha=ai, Re=3e6) for ai in a]
    xu = [r.x_tr_up for r in rs]
    xl = [r.x_tr_lo for r in rs]
    mu = [r.tr_mode_up for r in rs]
    ml = [r.tr_mode_lo for r in rs]
    NUM["modes_up"] = dict(zip(map(float, a), mu))
    NUM["modes_lo"] = dict(zip(map(float, a), ml))
    sep = [(ai, x) for ai, x, m_ in zip(a, xl, ml) if m_ == "laminar_sep"] + \
          [(ai, x) for ai, x, m_ in zip(a, xu, mu) if m_ == "laminar_sep"]
    sepc = co([s[0] for s in sep], [s[1] for s in sep]) if sep else ""
    write("xtr_alpha", rf"""\begin{{tikzpicture}}
\begin{{axis}}[width=0.7\linewidth,height=5.6cm,xlabel={{$\alpha$ (°)}},ylabel={{$x_{{tr}}/c$}},xmin=-4,xmax=10,ymin=0,ymax=1,
  legend style={{draw=none,font=\scriptsize,at={{(0.02,0.02)}},anchor=south west}}]
\addplot[cblue,very thick,mark=*,mark size=1.4pt] coordinates {{{co(a, xu)}}};\addlegendentry{{extrados}}
\addplot[corange,very thick,mark=square*,mark size=1.4pt] coordinates {{{co(a, xl)}}};\addlegendentry{{intrados}}
{"\\addplot[only marks,mark=o,mark size=3.5pt,cink,thick] coordinates {" + sepc + "};\\addlegendentry{décollement laminaire}" if sep else ""}
\end{{axis}}
\end{{tikzpicture}}""")


def fig_xtrip():
    xt = np.linspace(0.02, 0.8, 27)
    cd = [evaluate(0.02, 0.4, 0.12, alpha=2.0, Re=3e6, x_trip=(x, x), free_transition=False).CD for x in xt]
    free = evaluate(0.02, 0.4, 0.12, alpha=2.0, Re=3e6)
    NUM.update(cd_trip_002=float(cd[0]), cd_trip_max=float(cd[-1]), cd_free=float(free.CD))
    write("cd_xtrip", rf"""\begin{{tikzpicture}}
\begin{{axis}}[width=0.7\linewidth,height=5.4cm,xlabel={{transition imposée $x_{{tr}}/c$ (deux faces)}},ylabel={{$C_D$ (counts)}},
  xmin=0,xmax=0.8,legend style={{draw=none,font=\scriptsize}},legend pos=north east]
\addplot[cblue,very thick,mark=*,mark size=1.2pt] coordinates {{{co(xt, 1e4*np.array(cd))}}};\addlegendentry{{transition imposée}}
\addplot[corange,dashed,very thick] coordinates {{(0,{1e4*free.CD:.2f}) (0.8,{1e4*free.CD:.2f})}};\addlegendentry{{transition libre (Michel)}}
\end{{axis}}
\end{{tikzpicture}}""")


def fig_te():
    Re = 3e6
    r, inv, _ = evaluate(0.0, 0.0, 0.12, alpha=0.0, Re=Re, return_details=True)
    up, lo = split_at_stagnation(inv)
    s, x, Ue, _ = up
    b1 = march_surface(s, x, Ue, Re, x_stop=1.0)
    j = b1.x > 0.85
    xs = np.array([0.95, 0.96, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995, 1.0])
    cds = [evaluate(0, 0, 0.12, alpha=0, Re=Re, x_stop=v).CD for v in xs]
    cds2 = [evaluate(0.02, 0.4, 0.12, alpha=2, Re=Re, x_stop=v).CD for v in xs]
    NUM.update(te_cd_0985=float(cds[5]), te_cd_1=float(cds[-1]), te_ue_0985=float(np.interp(0.985, b1.x, b1.Ue)),
               te_ue_last=float(b1.Ue[-1]), te_x_last=float(b1.x[-1]),
               te_th_0985=float(np.interp(0.985, b1.x, b1.theta)), te_th_last=float(b1.theta[-1]),
               te_cd2_0985=float(cds2[5]), te_cd2_1=float(cds2[-1]))
    write("bord_de_fuite", rf"""\begin{{tikzpicture}}
\begin{{groupplot}}[group style={{group size=3 by 1,horizontal sep=1.2cm}},width=0.35\linewidth,height=5cm,
  title style={{font=\small}},tick label style={{font=\scriptsize}},legend style={{draw=none,font=\scriptsize}}]
\nextgroupplot[title={{$U_e/U_\infty$ (NACA 0012, $\alpha=0$)}},xlabel={{$x/c$}},xmin=0.85,xmax=1]
\addplot[cblue,very thick,mark=*,mark size=1pt] coordinates {{{co(b1.x[j], b1.Ue[j])}}};
\addplot[cmuted,dashed] coordinates {{(0.985,0.6) (0.985,1.0)}};
\node[cmuted,font=\scriptsize,anchor=south east,rotate=90] at (axis cs:0.985,0.62) {{$x_{{stop}}$}};
\nextgroupplot[title={{$10^3\,\theta/c$}},xlabel={{$x/c$}},xmin=0.85,xmax=1]
\addplot[corange,very thick,mark=*,mark size=1pt] coordinates {{{co(b1.x[j], 1e3*b1.theta[j])}}};
\addplot[cmuted,dashed] coordinates {{(0.985,1) (0.985,4)}};
\nextgroupplot[title={{$C_D$ (counts) selon $x_{{stop}}$}},xlabel={{$x_{{stop}}$}},xmin=0.95,xmax=1,legend pos=north west]
\addplot[cblue,very thick,mark=*,mark size=1.3pt] coordinates {{{co(xs, 1e4*np.array(cds))}}};\addlegendentry{{0012, $\alpha=0$}}
\addplot[corange,very thick,mark=square*,mark size=1.3pt] coordinates {{{co(xs, 1e4*np.array(cds2))}}};\addlegendentry{{2412, $\alpha=2^\circ$}}
\end{{groupplot}}
\end{{tikzpicture}}""")


def fig_polaire():
    a = np.arange(-4, 11, 1.0)
    rs = [evaluate(0.02, 0.4, 0.12, alpha=ai, Re=3e6) for ai in a]
    CL = np.array([r.CL for r in rs]); CD = np.array([r.CD for r in rs]) * 1e4
    CF = np.array([r.CD_f for r in rs]) * 1e4
    ok = np.array([r.valid for r in rs])
    LD = CL / (CD / 1e4)
    ib = int(np.nanargmax(np.where(ok, LD, np.nan)))
    NUM.update(polar_LDmax=float(LD[ib]), polar_a_LDmax=float(a[ib]), polar_invalid=[float(x) for x in a[~ok]],
               polar_flags={float(ai): r.flags for ai, r in zip(a, rs) if r.flags})
    inval = co(CD[~ok], CL[~ok]) if (~ok).any() else ""
    write("polaire", rf"""\begin{{tikzpicture}}
\begin{{groupplot}}[group style={{group size=2 by 1,horizontal sep=1.6cm}},width=0.5\linewidth,height=5.6cm,
  title style={{font=\small}},tick label style={{font=\scriptsize}},legend style={{draw=none,font=\scriptsize}}]
\nextgroupplot[title={{Polaire $C_L(C_D)$}},xlabel={{$C_D$ (counts)}},ylabel={{$C_L$}},legend pos=south east]
\addplot[cblue,very thick,mark=*,mark size=1.3pt] coordinates {{{co(CD[ok], CL[ok])}}};\addlegendentry{{valide}}
{"\\addplot[only marks,mark=x,mark size=4pt,corange,very thick] coordinates {" + inval + "};\\addlegendentry{invalide}" if inval else ""}
\nextgroupplot[title={{Décomposition de $C_D$}},xlabel={{$\alpha$ (°)}},ylabel={{counts}},legend pos=north west,ymin=0]
\addplot[cblue,very thick] coordinates {{{co(a, CD)}}};\addlegendentry{{$C_D$ total (Squire-Young)}}
\addplot[cgreen,very thick] coordinates {{{co(a, CF)}}};\addlegendentry{{frottement $C_{{D,f}}$}}
\addplot[corange,very thick] coordinates {{{co(a, CD - CF)}}};\addlegendentry{{forme $C_D-C_{{D,f}}$}}
\end{{groupplot}}
\end{{tikzpicture}}""")


# ===================================================================== base et QUBO
def load(path):
    rows = list(csv.DictReader(open(path)))
    return rows


def fig_base():
    A = load(os.path.join(CODE, "resultats", "base_aero_Re3e+06_a2.csv"))
    C = load(os.path.join(CODE, "resultats", "base_aero_Re3e+06_CL0.4.csv"))

    def grid(rows, key, ip=4, scale=1.0):
        out = []
        for r in rows:
            if int(r["ip"]) == ip:
                v = float(r[key]) * scale
                out.append((int(r["im"]), float(r["t"]) * 100, v, r["valid"] == "True"))
        return out

    def matrix(rows, key, title, scale, label):
        g = grid(rows, key, scale=scale)
        g.sort(key=lambda z: (z[1], z[0]))
        pts = " ".join(f"({im},{t:.0f}) [{v:.4g}]" for im, t, v, ok in g)
        bad = " ".join(f"({im},{t:.0f})" for im, t, v, ok in g if not ok)
        return rf"""\nextgroupplot[title={{{title}}},colorbar,colorbar style={{ylabel={{{label}}},width=2.5mm,tick label style={{font=\scriptsize}}}}]
\addplot[matrix plot*,mesh/cols=8,point meta=explicit] coordinates {{{pts}}};
{"\\addplot[only marks,mark=x,mark size=2.5pt,cink,thick] coordinates {" + bad + "};" if bad else ""}"""

    write("base_cartes", rf"""\begin{{tikzpicture}}
\begin{{groupplot}}[group style={{group size=2 by 1,horizontal sep=3.2cm}},width=0.36\linewidth,height=5.6cm,
  xlabel={{cambrure $m$ (\%)}},ylabel={{épaisseur $t$ (\%)}},xtick={{0,...,7}},ytick={{6,8,...,20}},
  enlargelimits=false,axis on top,title style={{font=\small}},tick label style={{font=\scriptsize}},
  colormap={{bl}}{{color=(white) color=(cblue!45) color=(cblue!85!black)}}]
{matrix(A, "CD", "$\\alpha=2^\\circ$ imposé", 1e4, "$C_D$ (counts)")}
{matrix(C, "CD", "$C_L=0{,}4$ imposé", 1e4, "$C_D$ (counts)")}
\end{{groupplot}}
\end{{tikzpicture}}""")

    def stats(rows, tag):
        CD = np.array([float(r["CD"]) for r in rows])
        val = np.array([r["valid"] == "True" for r in rows])
        feas = np.array([all(float(g) <= 0 for g in r["g"].split(";")) for r in rows])
        m = np.array([float(r["m"]) for r in rows]); t = np.array([float(r["t"]) for r in rows])
        alpha = np.array([float(r["alpha"]) for r in rows])
        i = np.nanargmin(np.where(feas, CD, np.nan))
        NUM[tag] = dict(n_valid=int(val.sum()), n_feas=int(feas.sum()),
                        cd_min=float(np.nanmin(CD[val]) * 1e4), cd_max=float(np.nanmax(CD[val]) * 1e4),
                        corr_t=float(np.corrcoef(CD[val], t[val])[0, 1]), corr_m=float(np.corrcoef(CD[val], m[val])[0, 1]),
                        best=dict(m=rows[i]["m"], p=rows[i]["p"], t=rows[i]["t"], CD=float(CD[i] * 1e4)),
                        alpha_min=float(alpha.min()), alpha_max=float(alpha.max()),
                        n_te_sep=int(sum("te_separation" in r["flags"] for r in rows)))
    stats(A, "base_alpha")
    stats(C, "base_cl")

    # modèle quadratique A sur la base alpha
    rows = [r for r in A if r["valid"] == "True"]
    I = np.array([[int(r["im"]), int(r["ip"]), int(r["it"])] for r in rows])
    CD = np.array([float(r["CD"]) for r in rows])
    z = I / 7 * 2 - 1
    F = np.column_stack([np.ones(len(z))] + [z[:, i] for i in range(3)] +
                        [z[:, i] * z[:, j] for i in range(3) for j in range(i, 3)])
    c, *_ = np.linalg.lstsq(F, CD, rcond=None)
    pred = F @ c
    r2 = 1 - np.sum((CD - pred) ** 2) / np.sum((CD - CD.mean()) ** 2)
    modes = np.array([r["tr_mode_up"] for r in rows])
    NUM["quad_r2"] = float(r2)
    pl = []
    for mode, st, lab in (("michel", "cblue", "transition extrados : Michel"),
                          ("laminar_sep", "corange", "transition extrados : décollement laminaire")):
        k = modes == mode
        if k.any():
            pl.append(f"\\addplot[only marks,mark=*,mark size=1.1pt,{st},fill opacity=0.6,draw opacity=0.6] coordinates {{{co(1e4*CD[k], 1e4*pred[k], '{:.4g}')}}};\\addlegendentry{{{lab}}}")
    lo_, hi_ = 1e4 * CD.min(), 1e4 * CD.max()
    write("quadratique", rf"""\begin{{tikzpicture}}
\begin{{axis}}[width=0.62\linewidth,height=6cm,xlabel={{$C_D$ du modèle aéro (counts)}},ylabel={{$C_D$ quadratique (counts)}},
  legend style={{draw=none,font=\scriptsize,at={{(0.02,0.98)}},anchor=north west}},legend cell align=left]
{chr(10).join(pl)}
\addplot[cmuted,dashed] coordinates {{({lo_:.1f},{lo_:.1f}) ({hi_:.1f},{hi_:.1f})}};
\node[font=\small,anchor=south east] at (rel axis cs:0.98,0.03) {{$R^2={r2:.3f}$}};
\end{{axis}}
\end{{tikzpicture}}""")


def timing():
    t = time.time()
    for _ in range(20):
        evaluate(0.02, 0.4, 0.12, alpha=2.0)
    NUM["ms_par_profil"] = (time.time() - t) / 20 * 1e3
    t = time.time()
    for _ in range(5):
        alpha_for_cl(0.02, 0.4, 0.12, 0.4)
    NUM["ms_alpha_for_cl"] = (time.time() - t) / 5 * 1e3
    NUM["alpha_cl04_2412"] = alpha_for_cl(0.02, 0.4, 0.12, 0.4)


if __name__ == "__main__":
    for f in (fig_profils, fig_parametres, fig_cosinus, fig_cp, fig_convergence, fig_cl_alpha, fig_vt,
              fig_thwaites_closure, fig_plaque, fig_transition, fig_bl_profiles, fig_head_closure,
              fig_xtr_alpha, fig_xtrip, fig_te, fig_polaire, fig_base, timing):
        f()
        print("ok", f.__name__)
    with open(os.path.join(FIG, "chiffres.json"), "w") as fh:
        json.dump(NUM, fh, indent=1, default=float)
    print(json.dumps(NUM, indent=1, default=float))
