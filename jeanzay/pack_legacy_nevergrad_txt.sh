#!/bin/bash
# Regroupe, dossier par dossier, les .txt par run des anciens runs nevergrad (format
# "runtime, score", n<=256) dans un seul runs_budget_50000.tar au meme endroit, pour
# liberer des inodes sur $WORK. Rien ne quitte le dossier : `tar xf runs_budget_50000.tar`
# restaure tout.
#
# Un dossier n'est traite que si :
#   - final_scores_budget_50000.csv ET mean_curve_budget_50000.csv existent
#     (donc jamais les dossiers n=512, qui n'en ont pas),
#   - le CSV contient exactement autant de runs que de .txt (sinon il est incomplet :
#     le dossier est laisse intact et liste dans dossiers_a_reexporter.txt).
#
# Usage (depuis la racine du repo) :
#   bash jeanzay/pack_legacy_nevergrad_txt.sh          # apercu, ne modifie rien
#   bash jeanzay/pack_legacy_nevergrad_txt.sh --apply  # fait le regroupement

APPLY=0
[ "$1" = "--apply" ] && APPLY=1

ROOT="results/nevergrad"
REPORT="dossiers_a_reexporter.txt"
: > "$REPORT"

n_dirs=0
n_files=0
n_skipped=0

while read -r d; do
  [ -f "$d/mean_curve_budget_50000.csv" ] || continue
  n_txt=$(find "$d" -maxdepth 1 -name '*_budget_50000_*.txt' | wc -l)
  [ "$n_txt" -gt 0 ] || continue
  n_csv=$(( $(wc -l < "$d/final_scores_budget_50000.csv") - 1 ))
  if [ "$n_csv" -ne "$n_txt" ]; then
    echo "$d csv=$n_csv txt=$n_txt" >> "$REPORT"
    n_skipped=$((n_skipped + 1))
    continue
  fi
  n_dirs=$((n_dirs + 1))
  n_files=$((n_files + n_txt))
  if [ "$APPLY" -eq 1 ]; then
    (cd "$d" && tar cf runs_budget_50000.tar *_budget_50000_*.txt && rm -f *_budget_50000_*.txt) \
      || echo "ECHEC: $d"
  fi
done < <(find "$ROOT" -name final_scores_budget_50000.csv -printf '%h\n')

if [ "$APPLY" -eq 1 ]; then
  echo "Fait : $n_dirs dossiers regroupes, $n_files fichiers .txt -> $n_dirs .tar"
else
  echo "Apercu : $n_dirs dossiers a regrouper, $n_files fichiers .txt -> $n_dirs .tar"
  echo "Relancer avec --apply pour le faire."
fi
echo "Dossiers ignores (CSV incomplet) : $n_skipped -> voir $REPORT"
