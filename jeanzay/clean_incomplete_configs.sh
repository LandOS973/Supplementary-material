#!/bin/bash
# Supprime les configs de results/config/ qui n'ont pas toutes leurs instances
# (grid coupee par un timeout, le quota...), SAUF :
#   - la config en cours d'ecriture par un job de grid encore actif
#     (dernier "[CONFIG ...]" de son outputs/expe_<jobid>.out),
#   - les configs ViennaRNA (__target...), qui n'ont pas les memes instances.
# Une instance compte si son dossier contient raw_scores.csv.
#
# Usage (depuis la racine du repo) :
#   bash jeanzay/clean_incomplete_configs.sh          # apercu, ne supprime rien
#   bash jeanzay/clean_incomplete_configs.sh --apply  # supprime

EXPECTED=56
APPLY=0
[ "$1" = "--apply" ] && APPLY=1

running=$(squeue --me -h -o %i 2>/dev/null | while read -r j; do
  f="outputs/expe_${j}.out"
  [ -f "$f" ] && grep '^\[CONFIG' "$f" | tail -1 | awk '{print $3}'
done | sort -u)

echo "Configs en cours d'ecriture (protegees) :"
echo "${running:-  (aucune)}" | sed 's/^/  /'

n_del=0
for d in results/config/k*/; do
  name=$(basename "$d")
  case "$name" in *__target*) continue;; esac
  if printf '%s\n' "$running" | grep -qxF "$name"; then continue; fi
  n=$(find "$d" -mindepth 2 -maxdepth 2 -name raw_scores.csv | wc -l)
  [ "$n" -ge "$EXPECTED" ] && continue
  echo "incomplete ($n/$EXPECTED) : $name"
  n_del=$((n_del + 1))
  [ "$APPLY" -eq 1 ] && rm -rf "$d"
done

if [ "$APPLY" -eq 1 ]; then echo "Supprimees : $n_del"; else echo "A supprimer : $n_del (relancer avec --apply)"; fi
