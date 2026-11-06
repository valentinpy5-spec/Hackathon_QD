"""Figures V2 du cours quantique : montée en taille de QAOA (lit Code/resultats/montee_en_taille.json).

Usage (depuis le dossier du cours) : python3 tools/make_figures_v2.py
"""
import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJ = os.path.abspath(os.path.join(HERE, "..", ".."))
D = json.load(open(os.path.join(PROJ, "Code", "resultats", "montee_en_taille.json")))
FIG = os.path.join(HERE, "figures")


def co(xs, ys):
    return " ".join(f"({x:.5g},{y:.5g})" for x, y in zip(xs, ys))


n = [r["n"] for r in D]
q = lambda p, k: [r["qaoa"][str(p)][k] for r in D]
body = rf"""\begin{{tikzpicture}}
\begin{{groupplot}}[group style={{group size=2 by 1,horizontal sep=1.6cm}},width=0.48\linewidth,height=5.8cm,
  xlabel={{nombre de qubits $n$}},xtick={{6,9,12,15,18}},title style={{font=\small}},tick label style={{font=\scriptsize}},
  legend style={{draw=none,font=\scriptsize}}]
\nextgroupplot[ymode=log,title={{probabilité de mesurer l'optimum}},legend pos=south west]
\addplot[cblue,very thick,mark=*,mark size=1.3pt] coordinates {{{co(n, q(1, 'p_opt'))}}};\addlegendentry{{QAOA $p=1$}}
\addplot[corange,very thick,mark=square*,mark size=1.3pt] coordinates {{{co(n, q(2, 'p_opt'))}}};\addlegendentry{{QAOA $p=2$}}
\addplot[cviolet,very thick,mark=triangle*,mark size=1.6pt] coordinates {{{co(n, q(4, 'p_opt'))}}};\addlegendentry{{QAOA $p=4$}}
\addplot[cmuted,dashed,thick] coordinates {{{co(n, [r['p_random'] for r in D])}}};\addlegendentry{{hasard, $2^{{-n}}$}}
\nextgroupplot[title={{qualité de l'échantillon}},ymin=0,ymax=1,legend pos=south west]
\addplot[corange,very thick,mark=square*,mark size=1.3pt] coordinates {{{co(n, q(2, 'p_top1'))}}};\addlegendentry{{$P$(1~\% meilleur), $p=2$}}
\addplot[cviolet,very thick,mark=triangle*,mark size=1.6pt] coordinates {{{co(n, q(4, 'p_top1'))}}};\addlegendentry{{$P$(1~\% meilleur), $p=4$}}
\addplot[cgreen,very thick,mark=*,mark size=1.3pt] coordinates {{{co(n, q(4, 'ratio'))}}};\addlegendentry{{rapport d'approx., $p=4$}}
\addplot[cmuted,very thick,dashed,mark=o,mark size=1.3pt] coordinates {{{co(n, [r['sa_success'] for r in D])}}};\addlegendentry{{recuit : lectures optimales}}
\end{{groupplot}}
\end{{tikzpicture}}"""
open(os.path.join(FIG, "v2_montee.tex"), "w").write("% figure générée par tools/make_figures_v2.py\n" + body + "\n")
print("ok")
