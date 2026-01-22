#!/usr/bin/env bash
set -euo pipefail

# ===== CONFIG =====
PKG_PATH="src/yourpkg/version.py"
DIST_DIR="dist"
TAG_PREFIX="v"

# ===== CHECK TOOLS =====
command -v git >/dev/null || { echo "git not found"; exit 1; }
command -v gh  >/dev/null || { echo "gh CLI not found"; exit 1; }

# ===== GET VERSION =====
VERSION=$(python - << EOF
import importlib.util, sys
spec = importlib.util.spec_from_file_location("version", "$PKG_PATH")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print(mod.__version__)
EOF
)

TAG="${TAG_PREFIX}${VERSION}"

echo "📦 Version: $VERSION"
echo "🏷️  Tag: $TAG"

# ===== GIT STATE =====
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "❌ Working tree is dirty"
  exit 1
fi

if git rev-parse "$TAG" >/dev/null 2>&1; then
  echo "❌ Tag $TAG already exists"
  exit 1
fi

# ===== BUILD =====
echo "🔨 Building wheel..."
rm -rf "$DIST_DIR"
uv build

WHEEL=$(ls "$DIST_DIR"/*.whl | head -n 1)
if [ -z "$WHEEL" ]; then
  echo "❌ Wheel not found"
  exit 1
fi

echo "📦 Built: $WHEEL"

# ===== TAG =====
git tag -a "$TAG" -m "Release $TAG"
git push origin "$TAG"

# ===== RELEASE =====
echo "🚀 Creating GitHub release..."
gh release create "$TAG" \
  "$WHEEL" \
  --title "$TAG" \
  --notes "Release $TAG"

echo "✅ Release $TAG created successfully"
