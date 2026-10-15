"""Show registered experiments and their key YAML settings without launching Isaac Sim.

    python scripts/list_experiments.py
    python scripts/list_experiments.py --env Isaac-Car-v1 --algorithm c2rl-ppo
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source/contractionRL"
TASKS = SOURCE / "contractionRL/tasks/direct"
KEY_ARGS = {
    "ppo": ("learning_rate", "discount_factor", "rollouts", "learning_epochs", "ratio_clip"),
    "sac": ("learning_rate", "discount_factor", "batch_size", "gradient_steps"),
    "c3m": ("lbd", "w_lb", "w_ub", "W_lr", "u_lr", "use_empirical_dynamics"),
    "lqr": ("Q_scaler", "R_scaler"),
    "sdlqr": ("Q_scaler", "R_scaler"),
    "cvstem-lqr": ("lbd", "cm_samples", "cm_solver", "cm_w_lb", "cm_w_ub"),
    "c2rl-ppo": ("cmg_method", "lbd", "w_lb", "w_ub", "discount_factor", "learning_rate"),
    "c2rl-sac": ("cmg_method", "lbd", "w_lb", "w_ub", "discount_factor", "learning_rate"),
}
CFG_CLASSES = {
    "c3m": ("c3m.py", "C3MCfg"),
    "lqr": ("sdlqr.py", "SDLQRCfg"),
    "sdlqr": ("sdlqr.py", "SDLQRCfg"),
    "cvstem-lqr": ("cvstem_lqr.py", "CVSTEMLQRCfg"),
    "c2rl-ppo": ("c2rl.py", "C2RLPPOCfg"),
    "c2rl-sac": ("c2rl.py", "C2RLSACCfg"),
}
METHODS = {"c3m": "C1/C2 joint metric/controller training",
           "cvstem-lqr": "CV-STEM metric + analytical feedback",
           "c2rl-ppo": "CV-STEM or CCM metric training + PPO",
           "c2rl-sac": "CV-STEM or CCM metric training + SAC"}


def _entry(node, package, imports):
    """Resolve registration strings without importing simulator packages."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant):
                parts.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                expr = value.value
                if isinstance(expr, ast.Name) and expr.id == "__name__":
                    parts.append(package)
                elif isinstance(expr, ast.Attribute) and expr.attr == "__name__":
                    parts.append(imports[expr.value.id])
                else:
                    raise ValueError(f"Unsupported registration string in {package}")
        return "".join(parts)
    raise ValueError(f"Unsupported registration in {package}")


def environments():
    records = {}
    aliases = []
    for path in sorted(TASKS.rglob("__init__.py")):
        package = ".".join(path.parent.relative_to(SOURCE).parts)
        tree = ast.parse(path.read_text())
        imports = {}
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                base = (importlib.util.resolve_name("." * node.level + (node.module or ""), package)
                        if node.level else node.module)
                for alias in node.names:
                    imports[alias.asname or alias.name] = f"{base}.{alias.name}"
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name) and node.func.id == "_register":
                aliases.append(tuple(ast.literal_eval(arg) for arg in node.args))
            kwargs = {kw.arg: kw.value for kw in node.keywords}
            if not isinstance(kwargs.get("id"), ast.Constant):
                continue
            task = kwargs["id"].value
            configs = {}
            for key, value in zip(kwargs["kwargs"].keys, kwargs["kwargs"].values):
                name = ast.literal_eval(key)
                if name.startswith("skrl_") and name != "skrl_cfg_entry_point":
                    module, filename = _entry(value, package, imports).split(":")
                    algo = name.removeprefix("skrl_").removesuffix("_cfg_entry_point").replace("_", "-")
                    configs[algo] = SOURCE.joinpath(*module.split(".")) / filename
            records[task] = {"id": task, "family": "classic" if task.startswith("classic-") else "Isaac",
                             "dynamics": "analytical" if task.startswith("classic-") else "PhysX",
                             "configs": configs}
    for task, classic_task, _ in aliases:
        records[task] = {**records[classic_task], "id": task, "family": "Isaac",
                         "dynamics": f"exact {classic_task}"}
    return sorted(records.values(), key=lambda item: (item["family"] != "classic", item["id"]))


def key_settings(algo, path):
    raw = yaml.safe_load(path.read_text())
    # Same precedence as ContractionRunner. Defaults are read from the source
    # dataclass so omitted settings still show the deployed values.
    merged = {}
    filename, cls_name = CFG_CLASSES.get(algo, (None, None))
    tree = ast.parse((SOURCE / "contractionRL/agents/skrl" / filename).read_text()) if filename else ast.Module(body=[], type_ignores=[])
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef) and cls.name == cls_name:
            for node in cls.body:
                if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
                    try:
                        merged[node.target.id] = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        pass
    for block in ("cm", "cmg", "empirical_dynamics", "agent"):
        merged.update(raw.get(block, {}))
    return {key: merged[key] for key in KEY_ARGS[algo] if key in merged}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", default="", help="environment ID or substring")
    parser.add_argument("--algorithm", choices=KEY_ARGS, help="show one algorithm")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()
    color = sys.stdout.isatty() and not args.no_color

    def title(text):
        print(f"\n\033[1;36m{text}\033[0m" if color else f"\n{text}")
        print("━" * 100)

    rows = [row for row in environments() if args.env.lower() in row["id"].lower()
            and (args.algorithm is None or args.algorithm in row["configs"])]
    if not rows:
        parser.error("No registered environments match these filters.")
    title("ALGORITHMS · key configuration arguments")
    for algo, keys in KEY_ARGS.items():
        if args.algorithm is None or args.algorithm == algo:
            print(f"  {algo:13}  {', '.join(keys)}")
            if algo in METHODS:
                print(f"                 {METHODS[algo]}")
    print("\n  cmg_method: cvstem = SDP regression; ccm = full Killing + projected contraction + bounded-input feasibility.")
    print("  Settings below come from each task's YAML and algorithm defaults; they are not CLI flags.")
    title(f"ENVIRONMENTS · {len(rows)} registered tasks")
    for row in rows:
        print(f"\n  {row['id']}  [{row['family']} · {row['dynamics']}]")
        if not args.env and args.algorithm is None:
            print("    " + ", ".join(algo for algo in KEY_ARGS if algo in row["configs"]))
            continue
        for algo in KEY_ARGS:
            if algo not in row["configs"] or (args.algorithm is not None and algo != args.algorithm):
                continue
            path = row["configs"][algo]
            settings = key_settings(algo, path)
            values = "  ".join(f"{key}={value}" for key, value in settings.items())
            print(f"    {algo:13} {values}")
            if args.env:
                print(f"                  {path.relative_to(ROOT)}")
    print()


if __name__ == "__main__":
    main()
