"""Regenerate paper figures and tables from the certified R&R data bundles.

The notebook's code cells are ordinary Python, so running them in one process
avoids requiring a Jupyter kernel or network sockets on a cluster node.
"""

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "paper_facing_plots.ipynb"
OUTPUT = ROOT / "data" / "new_outputs" / "paper_facing_plots"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT / ".matplotlib"))
    os.chdir(ROOT)
    notebook = json.loads(NOTEBOOK.read_text())
    namespace = {"__name__": "__main__"}
    count = 0
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        print(f"Running notebook cell {index}", flush=True)
        exec(compile(source, f"{NOTEBOOK}:cell{index}", "exec"), namespace)
        count += 1
    (OUTPUT / "render_manifest.json").write_text(json.dumps({
        "source_notebook": str(NOTEBOOK.relative_to(ROOT)),
        "code_cells_executed": count,
        "figures": sorted(p.name for p in (OUTPUT / "figures").iterdir() if p.is_file()),
        "tables": sorted(p.name for p in (OUTPUT / "tables").iterdir() if p.is_file()),
    }, indent=2) + "\n")
    print(f"Regenerated figures and tables in {OUTPUT}")


if __name__ == "__main__":
    main()
