#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from datetime import date
from pathlib import Path


VERSION_RE = re.compile(r'^_version_\s*=\s*["\']([^"\']+)["\']', re.M)
DATE_RE    = re.compile(r'^_release_date_\s*=\s*["\']([^"\']+)["\']', re.M)


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

    args = p.parse_args()

    bump_part = (
        "major" if args.major else
        "minor" if args.minor else
        "patch" if args.patch else
        None
    )

    update_version_file(
        args.file,
        new_version=args.set_version,
        bump_part=bump_part,
    )


if __name__ == "__main__":
    main()
