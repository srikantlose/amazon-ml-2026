"""Build <team_name>_submission.zip in the layout required by the organizers:

    <team_name>_submission.zip
    ├── output/{matching_results.tsv, candidate_pairs.tsv}
    ├── code/business_entity_resolution/{src/, configs/, README.md, requirements.txt}
    └── Documentation_template.md

    python scripts/make_submission_zip.py --team "TeamName" [--outputs submissions/07_xxx]

requirements.txt in the zip pins the exact versions installed in the current environment.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = "code/business_entity_resolution"


def pinned_requirements() -> str:
    wanted = {re.sub(r"[-_.]+", "-", line.split("#")[0].strip().lower())
              for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
              if line.split("#")[0].strip()}
    frozen = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True,
                            check=True).stdout.splitlines()
    keep = [f for f in frozen if re.sub(r"[-_.]+", "-", f.split("==")[0].lower()) in wanted]
    header = ("# Pinned from the environment that produced the submission (Python "
              f"{sys.version.split()[0]}).\n# torch is the CUDA 12.8 build: "
              "pip install torch --index-url https://download.pytorch.org/whl/cu128\n")
    return header + "\n".join(sorted(keep, key=str.lower)) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    ap.add_argument("--outputs", default=str(ROOT / "output"), help="folder holding the two TSVs")
    ap.add_argument("--doc", default=str(ROOT / "docs" / "Documentation_template.md"))
    ap.add_argument("--dest", default=str(ROOT.parent))
    args = ap.parse_args()

    team = re.sub(r"\s+", "_", args.team.strip())
    zpath = Path(args.dest) / f"{team}_submission.zip"
    outputs = Path(args.outputs)
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for name in ("matching_results.tsv", "candidate_pairs.tsv"):
            zf.write(outputs / name, f"output/{name}")
        for p in sorted((ROOT / "src").glob("*.py")):
            zf.write(p, f"{PKG}/src/{p.name}")
        for name in ("base.yaml", "stage2.yaml", "stage3.yaml", "stage2_big.yaml", "stage3_big.yaml",
                     "output_la.yaml", "output_lacat.yaml", "output_la_big.yaml", "output_lacat_big.yaml"):
            zf.write(ROOT / "configs" / name, f"{PKG}/configs/{name}")
        for name in ("reproduce_best.sh", "train_big.sh"):
            zf.write(ROOT / "scripts" / name, f"{PKG}/scripts/{name}")
        zf.write(ROOT / "docs" / "README_reproduce.md", f"{PKG}/README.md")
        zf.writestr(f"{PKG}/requirements.txt", pinned_requirements())
        zf.write(args.doc, "Documentation_template.md")
    print(f"wrote {zpath} ({zpath.stat().st_size / 1e6:.1f} MB)")
    with zipfile.ZipFile(zpath) as zf:
        for n in zf.namelist():
            print("  ", n)


if __name__ == "__main__":
    main()
