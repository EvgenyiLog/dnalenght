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


def update_changelog(
    path: Path,
    version: str,
    today: str,
    section: str | None = None,
    message: str | None = None,
) -> None:
    """Add entry to CHANGELOG.md in Keep a Changelog format."""
    # Create minimal changelog if missing
    if not path.exists():
        path.write_text(
            "# Changelog\n\n"
            "All notable changes to this project will be documented in this file.\n\n"
            "The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).\n\n"
        )
        print(f"✔ created {path.name}")

    content = path.read_text(encoding="utf-8")

    # Build entry
    entry_lines = [f"## [{version}] - {today}\n"]
    if message and section:
        entry_lines.extend([f"### {section}", f"- {message}", "\n"])
    elif message:
        entry_lines.extend([f"- {message}", "\n"])
    else:
        entry_lines.append("\n")

    entry = "\n".join(entry_lines)

    # Insert after first # Changelog header (case-insensitive)
    header_match = re.search(r'^(# Changelog|# Change Log|# CHANGELOG)\s*\n', content, re.IGNORECASE | re.MULTILINE)
    if header_match:
        # Find end of header block (first double newline after header)
        insert_pos = header_match.end()
        # Skip description paragraph(s) until we hit a ## header or end of meaningful content
        desc_end = re.search(r'\n\s*\n(?=\s*##\s|\s*$)', content[insert_pos:], re.MULTILINE)
        if desc_end:
            insert_pos += desc_end.end()
        content = content[:insert_pos] + entry + content[insert_pos:]
    else:
        # Prepend to file
        content = entry + content

    path.write_text(content, encoding="utf-8")
    print(f"✔ updated {path.name}: [{version}] {today}")


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

    # Changelog
    p.add_argument(
        "--changelog",
        type=Path,
        metavar="FILE",
        default=Path("CHANGELOG.md"),
        help="Path to changelog file (default: CHANGELOG.md)",
    )
    p.add_argument(
        "--create-changelog",
        action="store_true",
        help="Create changelog file if it doesn't exist",
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

    today = date.today().isoformat()

    # Update changelog if requested
    if args.tag:
        if args.changelog.exists() or args.create_changelog:
            update_changelog(
                args.changelog,
                version=new_ver,
                today=today,
                section=args.changelog_section if args.changelog_msg else None,
                message=args.changelog_msg,
            )
        elif not args.changelog.exists():
            print(f"ℹ skipping {args.changelog.name} (file not found, use --create-changelog to create)")

    # Create Git tag and commit
    if args.tag:
        tag_name = f"v{new_ver}"
        commit_msg = args.commit_msg or f"chore: release {new_ver}"
        tag_msg = args.tag_msg or f"Release {new_ver} ({today})"

        try:
            # Stage all changed files (version.py + CHANGELOG.md)
            subprocess.run(["git", "add", "."], check=True, capture_output=True)
            # Commit with custom message
            subprocess.run(["git", "commit", "-m", commit_msg], check=True, capture_output=True)
            print(f"✔ committed: {commit_msg}")

            # Create annotated tag
            subprocess.run(["git", "tag", "-a", tag_name, "-m", tag_msg], check=True, capture_output=True)
            print(f"✔ tag created: {tag_name}")

            # Push changes and tag
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
