"""
Range results/the beast/ dans une arbo calquee sur source_code/instances/ :

    the beast/<probleme>/<N>/<K|t>/<instance>/{fitness.csv, solutions.csv}

    nk_64_1_0_results/nk_64_1_0.txt_fitness.csv     -> nk/64/1/nk_64_1_0/fitness.csv
    puboi_evo_n_64_t_5_i_1_results/..._fitness.csv  -> qubo/64/5/puboi_evo_n_64_t_5_i_1/fitness.csv

Le scan est RECURSIF : on peut glisser-deposer un dossier "<instance>_results" (ou juste
ses deux CSV) n'importe ou sous "the beast/", puis relancer :

    python reorganize.py           # dry-run (montre ce qui serait deplace)
    python reorganize.py --apply   # execute
"""
import re
import shutil
import sys
from pathlib import Path

BEAST = Path(__file__).resolve().parent

NK_RE = re.compile(r"^nk_(?P<N>\d+)_(?P<K>\d+)(?:_(?P<D>\d+))?_(?P<i>\d+)$")
QUBO_RE = re.compile(r"^puboi_evo_n_(?P<N>\d+)_t_(?P<t>\d+)_i_(?P<i>\d+)$")


def target_dir(stem: str) -> Path | None:
    """Dossier canonique pour une instance nommee `stem` (sans suffixe _results)."""
    m = NK_RE.match(stem)
    if m:
        D = m.group("D")
        problem = "nk3" if (D and int(D) > 2) else "nk"
        return BEAST / problem / m.group("N") / m.group("K") / stem
    m = QUBO_RE.match(stem)
    if m:
        return BEAST / "qubo" / m.group("N") / m.group("t") / stem
    return None


def canon_csv_name(fname: str) -> str | None:
    if fname.endswith("_fitness.csv"):
        return "fitness.csv"
    if fname.endswith("_solutions.csv"):
        return "solutions.csv"
    return None


def file_stem(fname: str) -> str:
    """'nk_64_1_0.txt_fitness.csv' -> 'nk_64_1_0' ; '..._i_1.json_solutions.csv' -> '..._i_1'."""
    base = re.sub(r"_(fitness|solutions)\.csv$", "", fname)
    return re.sub(r"\.(txt|json)$", "", base)


def main(apply: bool) -> None:
    moves: list[tuple[Path, Path]] = []
    skipped: list[str] = []
    seen: set[Path] = set()
    rel = lambda p: p.relative_to(BEAST)

    # 1) dossiers "<instance>_results" (a n'importe quelle profondeur)
    for sub in sorted(BEAST.rglob("*_results")):
        if not sub.is_dir():
            continue
        stem = sub.name[: -len("_results")]
        tgt = target_dir(stem)
        if tgt is None:
            skipped.append(f"{rel(sub)}  (nom d'instance non reconnu)")
            continue
        for f in sorted(sub.iterdir()):
            if not f.is_file():
                continue
            cname = canon_csv_name(f.name)
            if cname is None:
                skipped.append(f"{rel(f)}  (fichier non reconnu)")
                continue
            moves.append((f, tgt / cname))
            seen.add(f.resolve())

    # 2) CSV isoles "*_fitness.csv" / "*_solutions.csv" deja mal ranges ou laches en vrac
    for f in sorted(BEAST.rglob("*.csv")):
        if f.resolve() in seen or not f.is_file():
            continue
        cname = canon_csv_name(f.name)
        if cname is None:
            continue
        tgt = target_dir(file_stem(f.name))
        if tgt is None:
            skipped.append(f"{rel(f)}  (nom d'instance non reconnu)")
            continue
        if f.parent == tgt and f.name == cname:
            continue                                  # deja au bon endroit
        moves.append((f, tgt / cname))

    print(f"{'APPLIQUE' if apply else 'DRY-RUN'} — {len(moves)} fichier(s)\n")
    for src, dst in moves:
        print(f"  {rel(src)}\n      -> {rel(dst)}")
        if apply:
            if dst.exists():
                print("      !! existe deja, ignore")
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))

    if apply:
        for d in sorted(BEAST.rglob("*_results"), reverse=True):
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()
                print(f"\n  rmdir {rel(d)}")

    if skipped:
        print("\nIgnores :")
        for s in skipped:
            print(f"  - {s}")

    if not apply:
        print("\n(relancer avec --apply pour executer)")


if __name__ == "__main__":
    main(apply="--apply" in sys.argv)
