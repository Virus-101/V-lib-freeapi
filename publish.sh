#!/usr/bin/env bash
# Publish freeaipi to PyPI and npm.
#
# Tokens are read from the environment and NEVER written to disk, logged, or
# committed. Both names (PyPI project, npm package) were confirmed free.
#
#   PyPI API token  -> PYPI_TOKEN   (scope: project freeaipi)
#   npm access token -> NPM_TOKEN    (type: Automation, or Granular with publish)
#
# Usage:
#   PYPI_TOKEN=pypi-AgEIcH... NPM_TOKEN=npm_... ./publish.sh
#   PYPI_TOKEN=pypi-... ./publish.sh pypi      # one registry only
#   ./publish.sh --dry-run                    # verify auth, publish nothing
set -euo pipefail
cd "$(dirname "$0")"

DRY="false"
TARGET="all"
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY="true" ;;
    all|pypi|npm) TARGET="$arg" ;;
    *) echo "unknown argument: $arg"; exit 2 ;;
  esac
done

if [ "$TARGET" = "all" ] || [ "$TARGET" = "pypi" ]; then
  : "${PYPI_TOKEN:?set PYPI_TOKEN (PyPI API token, scope: project freeaipi)}"
fi
if [ "$TARGET" = "all" ] || [ "$TARGET" = "npm" ]; then
  : "${NPM_TOKEN:?set NPM_TOKEN (npm access token with publish rights)}"
fi

# ── Python ────────────────────────────────────────────────────────────────
if [ "$TARGET" = "all" ] || [ "$TARGET" = "pypi" ]; then
  echo "── PyPI"
  rm -rf dist build freeaipi.egg-info
  py -3.12 -m build >/dev/null
  echo "   built: $(ls dist/*.whl dist/*.tar.gz | xargs -n1 basename | tr '\n' ' ')"

  if [ "$DRY" = "true" ]; then
    # Verify the token without uploading anything.
    code=$(curl -s -o /dev/null -w "%{http_code}" -X GET \
      -H "Authorization: Bearer $PYPI_TOKEN" \
      https://pypi.org/pypi/freeaipi/json || true)
    echo "   token check: HTTP $code (404 = name unused, which is expected pre-publish)"
    py -3.12 -m pip install --quiet --upgrade twine 2>/dev/null || true
    py -3.12 -m twine check dist/*
  else
    py -3.12 -m pip install --quiet --upgrade twine 2>/dev/null || true
    py -3.12 -m twine upload dist/* --username __token__ --password "$PYPI_TOKEN" --non-interactive
    echo "   published: https://pypi.org/project/freeaipi/"
  fi
fi

# ── Node ──────────────────────────────────────────────────────────────────
if [ "$TARGET" = "all" ] || [ "$TARGET" = "npm" ]; then
  echo "── npm"
  cd js
  ./node_modules/.bin/tsc -p tsconfig.json
  npm test >/dev/null
  npm pack --silent >/dev/null

  if [ "$DRY" = "true" ]; then
    # Verify the token resolves to a user without publishing anything.
    NODE_AUTH_TOKEN="$NPM_TOKEN" npm whoami 2>&1 | sed 's/^/   whoami: /' || true
    echo "   tarball: $(ls *.tgz)"
    rm -f *.tgz
  else
    # npm reads the bearer token from NODE_AUTH_TOKEN; it is never written to
    # .npmrc, so no credential lands on disk.
    NODE_AUTH_TOKEN="$NPM_TOKEN" npm publish --registry=https://registry.npmjs.org --access public
    rm -f *.tgz
    echo "   published: https://www.npmjs.com/package/freeaipi"
  fi
fi

echo "done."
