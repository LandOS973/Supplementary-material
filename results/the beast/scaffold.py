"""
Cree l'arborescence VIDE de "the beast/" a partir de source_code/instances/,
un dossier par instance, pret pour glisser-deposer les CSV (fitness.csv / solutions.csv).

    the beast/<probleme>/<N>/<K|t>/<instance>/

Usage :
    python scaffold.py            # dry-run
    python scaffold.py --apply
"""
import re
import sys
from pathlib import Path

BEAST = Path(__file__).resolve().parent
INSTANCES = BEAST.parents[1] / "source_code" / "instances"

NK_RE = re.compile(r"^nk_(?P<N>\d+)_(?P<K>\d+)(?:_(?P<D>\d+))?_(?P<i>\d+)\.txt$")
QUBO_RE = re.compile(r"^puboi_evo_n_(?P<N>\d+)_t_(?P<t>\d+)_i_(?P<i>\d+)\.json$")


def leaves() -> list[Path]:
    out: list[Path] = []
    for f in INSTANCES.rglob("*.txt"):
        m = NK_RE.match(f.name)
        if not m:
            continue
        D = m.group("D")
        problem = "nk3" if (D and int(D) > 2) else "nk"
        stem = f.name[:-4]
        out.append(BEAST / problem / m.group("N") / m.group("K") / stem)
    for f in INSTANCES.rglob("*.json"):
        m = QUBO_RE.match(f.name)
        if not m:
            continue
        stem = f.name[:-5]
        out.append(BEAST / "qubo" / m.group("N") / m.group("t") / stem)
    return sorted(set(out))


def main(apply: bool) -> None:
    dirs = leaves()
    todo = [d for d in dirs if not d.exists()]
    print(f"{'APPLIQUE' if apply else 'DRY-RUN'} — {len(dirs)} instances, "
          f"{len(todo)} dossier(s) a creer\n")
    for d in todo:
        print(f"  + {d.relative_to(BEAST)}")
        if apply:
            d.mkdir(parents=True, exist_ok=True)
    if not apply:
        print("\n(relancer avec --apply pour creer)")


if __name__ == "__main__":
    main(apply="--apply" in sys.argv)
