#!/usr/bin/env bash
# Recrée les outils locaux du projet, sans droits administrateur :
#   .venv/   environnement Python avec pymoo (hérite de NumPy/SciPy du système)
#   .tools/  XFOIL et l'écran virtuel Xvfb, extraits des paquets Debian
# Ces dossiers ne sont pas suivis par git (binaires) : ce script suffit à les reconstruire.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  python3 -m venv --system-site-packages .venv
  .venv/bin/pip install -q pymoo
fi

mkdir -p .tools
cd .tools
for pkg in xfoil xvfb; do
  if [ ! -d "${pkg}_pkg" ]; then
    apt-get download "$pkg"
    dpkg -x ${pkg}_*.deb "${pkg}_pkg"
  fi
done
echo "Outils prêts : .venv (pymoo), .tools/xfoil_pkg, .tools/xvfb_pkg"
