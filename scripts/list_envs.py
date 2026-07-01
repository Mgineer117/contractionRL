"""List all registered environments in the contractionRL extension.

Works without Isaac Sim by scanning task __init__.py files directly.

Usage:
    python scripts/list_envs.py
    python scripts/list_envs.py --keyword quadruped
    python scripts/list_envs.py --keyword path_tracking
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


def _find_task_roots() -> Path:
    """Locate the tasks/direct directory relative to this script."""
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    return repo_root / "source" / "contractionRL" / "contractionRL" / "tasks" / "direct"


def _parse_registrations(tasks_dir: Path) -> list[dict]:
    """ 
    Walk every __init__.py under tasks_dir and extract gym.register calls.
    Returns list of dicts with keys: id, entry_point, env_cfg, skrl_cfgs, pkg_dir.
    """
    envs = []
    for init_path in sorted(tasks_dir.rglob("__init__.py")):
        if "agents" in init_path.parts:
            continue
        text = init_path.read_text()
        if "gym.register" not in text:
            continue

        for block in re.findall(r"gym\.register\s*\((.+?)\)", text, re.DOTALL):
            env_id = _extract(r'id\s*=\s*["\']([^"\']+)["\']', block)
            if not env_id:
                continue

            # entry_point may be a plain string or an f-string like f"{__name__}.env:ClassName"
            # — we just want the trailing Class name after the colon
            entry_point = (
                _extract(r'entry_point\s*=\s*f?["\'][^"\']*:([A-Za-z_][A-Za-z0-9_]*)["\']', block)
                or _extract(r'entry_point\s*=\s*f?["\']([^"\']+)["\']', block)
                or "-"
            )

            # env_cfg entry_point — same pattern
            env_cfg = (
                _extract(r'"env_cfg_entry_point"\s*:\s*f?["\'][^"\']*:([A-Za-z_][A-Za-z0-9_]*)["\']', block)
                or _extract(r'"env_cfg_entry_point"\s*:\s*f?["\']([^"\']+)["\']', block)
                or "-"
            )

            # config entry points (skrl for Isaac envs, mjrl for classic envs)
            # — extract the YAML filename after the colon
            skrl_cfgs = {}
            for key, val in re.findall(
                r'"((?:skrl|mjrl)[^"]*_entry_point)"\s*:\s*f?["\'][^"\']*:([^\s"\']+)["\']', block
            ):
                skrl_cfgs[key] = val   # e.g. "skrl_ppo_cfg.yaml" / "mjrl_c3m_cfg.yaml"

            envs.append(
                {
                    "id":          env_id,
                    "entry_point": entry_point,
                    "env_cfg":     env_cfg,
                    "skrl_cfgs":   skrl_cfgs,
                    "pkg_dir":     str(init_path.parent.relative_to(tasks_dir)),
                }
            )
    return envs


def _extract(pattern: str, text: str) -> str | None:
    m = re.search(pattern, text, re.DOTALL)
    return m.group(1).strip() if m else None


def _col(text: str, width: int) -> str:
    return text[:width].ljust(width)


def main():
    parser = argparse.ArgumentParser(description="List contractionRL environments.")
    parser.add_argument("--keyword", type=str, default=None, help="Filter by keyword.")
    args = parser.parse_args()

    tasks_dir = _find_task_roots()
    if not tasks_dir.exists():
        print(f"[ERROR] tasks/direct directory not found: {tasks_dir}")
        return

    envs = _parse_registrations(tasks_dir)

    if args.keyword:
        envs = [e for e in envs if args.keyword.lower() in e["id"].lower()]

    if not envs:
        print("No environments found" + (f" matching '{args.keyword}'" if args.keyword else "") + ".")
        return

    # column widths
    W = {"no": 4, "id": 50, "dir": 30, "cfg": 35}

    header = (
        f"{'No.':>{W['no']}}  "
        f"{'Task ID':<{W['id']}}  "
        f"{'Directory':<{W['dir']}}  "
        f"{'skrl configs':<{W['cfg']}}"
    )
    sep = "-" * len(header)

    print()
    print(f"  contractionRL — {len(envs)} environment(s)"
          + (f"  [filter: '{args.keyword}']" if args.keyword else ""))
    print(sep)
    print(header)
    print(sep)

    for i, e in enumerate(envs, 1):
        cfgs = ", ".join(
            k.replace("skrl_", "").replace("_cfg_entry_point", "").replace("_entry_point", "")
            for k in e["skrl_cfgs"]
        ) or "-"
        print(
            f"{i:>{W['no']}}  "
            f"{_col(e['id'], W['id'])}  "
            f"{_col(e['pkg_dir'], W['dir'])}  "
            f"{cfgs}"
        )

    print(sep)
    print()

    # per-env detail
    print("Detail")
    print(sep)
    for e in envs:
        print(f"  {e['id']}")
        # shorten entry_point for readability
        ep = e["entry_point"]
        # collapse the long package prefix to just module.Class
        ep_short = re.sub(r".*\.tasks\.direct\.", "", ep)
        print(f"    env        : {ep_short}")
        print(f"    env_cfg    : {e['env_cfg'].split('.')[-1] if e['env_cfg'] != '-' else '-'}")
        for k, v in e["skrl_cfgs"].items():
            label = k.replace("skrl_", "").replace("_cfg_entry_point", "").replace("_entry_point", "")
            print(f"    skrl/{label:<8}: {v.split(':')[-1]}")
        print()


if __name__ == "__main__":
    main()
