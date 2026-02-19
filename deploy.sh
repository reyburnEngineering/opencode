#!/bin/bash
set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_PATH="$HOME/.opencode/bin/opencode"
PKG_DIR="$REPO_DIR/packages/opencode"

echo "Building opencode from source..."
cd "$PKG_DIR"
bun run build -- --single

DIST_DIR=$(ls -d "$PKG_DIR/dist"/opencode-* 2>/dev/null | head -1)
if [ -z "$DIST_DIR" ]; then
  echo "Build failed: no dist output found"
  exit 1
fi

BINARY="$DIST_DIR/bin/opencode"
if [ ! -f "$BINARY" ]; then
  echo "Build failed: binary not found at $BINARY"
  exit 1
fi

cp "$BINARY" "$INSTALL_PATH"
echo "Deployed to $INSTALL_PATH"
opencode --version
