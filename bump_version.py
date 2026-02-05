#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from datetime import date
from pathlib import Path
import subprocess


VERSION_RE = re.compile(r'^_version_\s*=\s*["\']([^"\']+)["\']', re.M)
DATE_RE = re.compile(r'^_release_date_\s*=\s*["\']([^"\']+)["\']', re.M)
PYPROJECT_VERSION_RE = re.compile(r'^\s*version\s*=\s*["\'](\d+\.\d+\.\d+)["\']', re.M)


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
) -> str:
    """Update version.py and return new version string."""
    text = path.read_text(encoding="utf-8")

    m = VERSION_RE.search(text)
    if not m:
        raise RuntimeError(f"_version_ not found in {path}")

    old_version = m.group(1)

    if bump_part:
        new_version = bump(old_version, bump_part)
    elif not new_version:
        raise RuntimeError("No version change requested")

    today = date.today().isoformat()

    text = VERSION_RE.sub(f'_version_ = "{new_version}"', text)
    text = DATE_RE.sub(f'_release_date_ = "{today}"', text)

    path.write_text(text, encoding="utf-8")

    print(f"✔ version.py: {old_version} → {new_version}")
    print(f"✔ release date: {today}")
    return new_version


def update_pyproject_version(path: Path, new_version: str) -> None:
    """Update version field in pyproject.toml (PEP 621 format)."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    updated = False

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            continue
        
        m = PYPROJECT_VERSION_RE.match(line)
        if m:
            old_version = m.group(1)
            indent = line[:line.find('version')]
            quote = '"' if '"' in line else "'"
            lines[i] = f'{indent}version = {quote}{new_version}{quote}'
            print(f"✔ pyproject.toml: {old_version} → {new_version}")
            updated = True
            break

    if not updated:
        raise RuntimeError(
            "version field not found in pyproject.toml. "
            "Expected format: `version = \"X.Y.Z\"` inside [project] section."
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_changelog(
    path: Path,
    version: str,
    today: str,
    section: str | None = None,
    message: str | None = None,
) -> None:
    """Add entry to CHANGELOG.md in Keep a Changelog format."""
    if not path.exists():
        path.write_text(
            "# Changelog\n\n"
            "All notable changes to this project will be documented in this file.\n\n"
            "The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).\n\n"
        )
        print(f"✔ created {path.name}")

    content = path.read_text(encoding="utf-8")
    entry_lines = [f"## [{version}] - {today}\n"]
    if message and section:
        entry_lines.extend([f"### {section}", f"- {message}", "\n"])
    elif message:
        entry_lines.extend([f"- {message}", "\n"])
    else:
        entry_lines.append("\n")

    entry = "\n".join(entry_lines)
    header_match = re.search(r'^(# Changelog|# Change Log|# CHANGELOG)\s*\n', content, re.IGNORECASE | re.MULTILINE)
    
    if header_match:
        insert_pos = header_match.end()
        desc_end = re.search(r'\n\s*\n(?=\s*##\s|\s*$)', content[insert_pos:], re.MULTILINE)
        if desc_end:
            insert_pos += desc_end.end()
        content = content[:insert_pos] + entry + content[insert_pos:]
    else:
        content = entry + content

    path.write_text(content, encoding="utf-8")
    print(f"✔ updated {path.name}: [{version}] {today}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Bump version in version.py (+ auto-update pyproject.toml & CHANGELOG.md)"
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

    # Smart defaults with opt-out
    p.add_argument(
        "--no-pyproject",
        action="store_true",
        help="Skip updating pyproject.toml (default: update if exists)",
    )
    p.add_argument(
        "--no-changelog",
        action="store_true",
        help="Skip updating CHANGELOG.md (default: update on --tag if exists)",
    )
    p.add_argument(
        "--create-changelog",
        action="store_true",
        help="Create CHANGELOG.md if it doesn't exist",
    )
    p.add_argument(
        "--changelog-section",
        choices=["Added", "Changed", "Deprecated", "Removed", "Fixed", "Security"],
        default="Changed",
        help="Section for changelog entry (default: Changed)",
    )
    p.add_argument(
        "--changelog-msg",
        metavar="TEXT",
        help="Message for changelog entry (optional)",
    )

    # Git integration
    p.add_argument(
        "--tag",
        action="store_true",
        help="Create annotated Git tag and commit after version bump",
    )
    p.add_argument(
        "--commit-msg",
        metavar="TEXT",
        help="Custom commit message (default: 'chore: release X.Y.Z')",
    )
    p.add_argument(
        "--tag-msg",
        metavar="TEXT",
        help="Custom tag annotation message (default: 'Release X.Y.Z (YYYY-MM-DD)')",
    )

    args = p.parse_args()

    bump_part = (
        "major" if args.major else
        "minor" if args.minor else
        "patch" if args.patch else
        None
    )

    # Шаг 1: обновляем версию в version.py
    new_ver = update_version_file(
        args.file,
        new_version=args.set_version,
        bump_part=bump_part,
    )

    today = date.today().isoformat()

    # Шаг 2: обновляем pyproject.toml (по умолчанию, если не отключено и файл существует)
    pyproject_path = Path("pyproject.toml")
    if not args.no_pyproject and pyproject_path.exists():
        update_pyproject_version(pyproject_path, new_ver)
    elif not args.no_pyproject:
        print(f"ℹ skipping pyproject.toml (file not found)")
    else:
        print(f"ℹ skipping pyproject.toml (--no-pyproject)")

    # Шаг 3: обновляем changelog при --tag (по умолчанию, если не отключено)
    changelog_path = Path("CHANGELOG.md")
    if args.tag and not args.no_changelog:
        if changelog_path.exists() or args.create_changelog:
            update_changelog(
                changelog_path,
                version=new_ver,
                today=today,
                section=args.changelog_section if args.changelog_msg else None,
                message=args.changelog_msg,
            )
        else:
            print(f"ℹ skipping CHANGELOG.md (file not found, use --create-changelog to create)")
    elif args.tag and args.no_changelog:
        print(f"ℹ skipping CHANGELOG.md (--no-changelog)")

    # Шаг 4: коммит и тег
    if args.tag:
        tag_name = f"v{new_ver}"
        commit_msg = args.commit_msg or f"chore: release {new_ver}"
        tag_msg = args.tag_msg or f"Release {new_ver} ({today})"

        try:
            subprocess.run(["git", "add", "."], check=True, capture_output=True)
            subprocess.run(["git", "commit", "-m", commit_msg], check=True, capture_output=True)
            print(f"✔ committed: {commit_msg}")

            subprocess.run(["git", "tag", "-a", tag_name, "-m", tag_msg], check=True, capture_output=True)
            print(f"✔ tag created: {tag_name}")

            subprocess.run(["git", "push"], check=True, capture_output=True)
            subprocess.run(["git", "push", "origin", tag_name], check=True, capture_output=True)
            print(f"✔ tag pushed: {tag_name}")

        except subprocess.CalledProcessError as e:
            cmd = e.cmd[0] if e.cmd else "unknown"
            stderr = e.stderr.decode().strip() if e.stderr else ""
            stdout = e.stdout.decode().strip() if e.stdout else ""
            print(f"✘ git error ({cmd}): {stderr or stdout}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
