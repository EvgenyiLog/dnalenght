#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from datetime import date
from pathlib import Path
import subprocess
import sys
import datetime


VERSION_RE = re.compile(r'^_version_\s*=\s*["\']([^"\']+)["\']', re.M)
DATE_RE = re.compile(r'^_release_date_\s*=\s*["\']([^"\']+)["\']', re.M)


def bump(version: str, part: str) -> str:
    major, minor, patch = map(int, version.split("."))
    if part == "major":
        return f"{major+1}.0.0"
    if part == "minor":
        return f"{major}.{minor+1}.0"
    if part == "patch":
        return f"{major}.{minor}.{patch+1}"
    raise ValueError(part)


def update_version_file(
    path: Path,
    new_version: str | None,
    bump_part: str | None,
) -> None:
    text = path.read_text(encoding="utf-8")

    m = VERSION_RE.search(text)
    if not m:
        raise RuntimeError("_version_ not found")

    old_version = m.group(1)

    if bump_part:
        new_version = bump(old_version, bump_part)
    elif not new_version:
        raise RuntimeError("No version change requested")

    today = date.today().isoformat()

    text = VERSION_RE.sub(f'_version_ = "{new_version}"', text)
    text = DATE_RE.sub(f'_release_date_ = "{today}"', text)

    path.write_text(text, encoding="utf-8")

    print(f"✔ version: {old_version} → {new_version}")
    print(f"✔ release date: {today}")
    return new_version


def main() -> None:
    p = argparse.ArgumentParser(
        description="Update _version_ and _release_date_ in version.py"
    )
    p.add_argument(
        "file",
        type=Path,
        help="Path to version.py",
    )

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--patch", action="store_true", help="Bump patch version")
    g.add_argument("--minor", action="store_true", help="Bump minor version")
    g.add_argument("--major", action="store_true", help="Bump major version")
    g.add_argument("--set-version", metavar="X.Y.Z", help="Set exact version")
    g.add_argument("--tag", action="store_true",
                   help="Create and push annotated Git tag after version bump")

    args = p.parse_args()

    bump_part = (
        "major" if args.major else
        "minor" if args.minor else
        "patch" if args.patch else
        None
    )

    new_ver = update_version_file(
        args.file,
        new_version=args.set_version,
        bump_part=bump_part,
    )

    # === НОВЫЙ БЛОК: создание тега ===
    if args.tag:

        tag_name = f"v{new_ver}"
        commit_msg = f"chore: release {new_ver}"
        tag_msg = f"Release {new_ver} ({datetime.date.today().isoformat()})"

        try:
            # Фиксируем изменения
            subprocess.run(["git", "commit", "-am", commit_msg], check=True, capture_output=True)
            print(f"✔ committed: {commit_msg}")

            # Создаём аннотированный тег
            subprocess.run(["git", "tag", "-a", tag_name, "-m", tag_msg], check=True, capture_output=True)
            print(f"✔ tag created: {tag_name}")

            # Отправляем
            subprocess.run(["git", "push"], check=True, capture_output=True)
            subprocess.run(["git", "push", "origin", tag_name], check=True, capture_output=True)
            print(f"✔ tag pushed: {tag_name}")

        except subprocess.CalledProcessError as e:
            print(f"✘ git error ({e.cmd[0]}): {e.stderr.decode().strip() or e.stdout.decode().strip()}")
            sys.exit(1)


if __name__ == "__main__":
    main()
