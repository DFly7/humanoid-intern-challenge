"""Create a project-local LIBERO config without upstream's interactive import."""

import argparse
from pathlib import Path

import yaml

parser = argparse.ArgumentParser()
parser.add_argument("checkout", type=Path)
parser.add_argument("--config-dir", type=Path, default=Path(".libero"))
args = parser.parse_args()
root = args.checkout.resolve() / "libero/libero"
if not (root / "bddl_files").is_dir():
    raise SystemExit("Expected a LIBERO checkout containing libero/libero/bddl_files")
config = args.config_dir.resolve()
config.mkdir(parents=True, exist_ok=True)
dataset = args.checkout.resolve() / "libero/datasets"
dataset.mkdir(exist_ok=True)
paths = {
    "benchmark_root": str(root),
    "bddl_files": str(root / "bddl_files"),
    "init_states": str(root / "init_files"),
    "assets": str(root / "assets"),
    "datasets": str(dataset),
}
with (config / "config.yaml").open("x") as stream:
    yaml.safe_dump(paths, stream)
print(f"export LIBERO_CONFIG_PATH='{config}'")
