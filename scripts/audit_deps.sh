#!/usr/bin/env bash
#
# Local dependency audit — parity with .github/workflows/audit.yml
#
# Same engine, same pinned tool version as CI: osv-scanner reading uv.lock
# directly. The pinned version + checksum live in ONE place, .github/osv-scanner.pin,
# which both this script and the CI workflow read — so they can't drift, and a
# bump is a single deliberate edit there.
#
# Deliberately NOT pip-audit: it shells out to pip/ensurepip, which uv-managed
# Pythons don't ship, and silently audits the wrong environment when that fails
# — a hollow green check. osv-scanner reads the lockfile; nothing to mis-target.
#
# The osv-scanner binary is downloaded pinned and checksum-verified (fail-closed):
# a tampered or substituted release aborts the run rather than executing. Note
# that pinning the binary does NOT make findings stale — OSV vulnerability data
# is queried live at scan time.
#
# What this checks:
#   1. uv.lock is in sync with the pyproject files (no drift).
#   2. Locked package versions against the OSV vulnerability database.
#
# Usage:  scripts/audit_deps.sh        (from anywhere; resolves repo root itself)
# Supported host arch: linux x86_64 (matches CI and this project's dev machine).

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

die() { echo "ERROR: $*" >&2; exit 1; }

# Resolve the pinned, checksum-verified osv-scanner binary into OSV_BIN.
# Caches per-version under ~/.cache so repeat runs don't re-download.
ensure_osv_scanner() {
  [[ "$(uname -s)/$(uname -m)" == "Linux/x86_64" ]] \
    || die "this script pins the linux_amd64 binary; on $(uname -s)/$(uname -m) install osv-scanner via your package manager and run: osv-scanner scan --lockfile=uv.lock"

  # shellcheck source=/dev/null
  source "${repo_root}/.github/osv-scanner.pin"
  [[ "$SHA256_LINUX_AMD64" != PLACEHOLDER* ]] \
    || die "osv-scanner checksum not pinned. Fill SHA256_LINUX_AMD64 in .github/osv-scanner.pin from a trusted machine (see that file's header)."

  local cache_dir="${HOME}/.cache/osv-scanner"
  local bin="${cache_dir}/osv-scanner-${VERSION}"
  if [[ -x "$bin" ]] && echo "${SHA256_LINUX_AMD64}  ${bin}" | sha256sum -c --status -; then
    OSV_BIN="$bin"; return
  fi

  mkdir -p "$cache_dir"
  echo "==> Fetching pinned osv-scanner v${VERSION}"
  curl -sSfL "https://github.com/google/osv-scanner/releases/download/v${VERSION}/osv-scanner_linux_amd64" -o "${bin}.tmp"
  if ! echo "${SHA256_LINUX_AMD64}  ${bin}.tmp" | sha256sum -c --status -; then
    rm -f "${bin}.tmp"
    die "osv-scanner checksum mismatch for v${VERSION}. If you just bumped VERSION, update the hash in .github/osv-scanner.pin. If you changed nothing, treat this as a possibly tampered download."
  fi
  chmod +x "${bin}.tmp" && mv "${bin}.tmp" "$bin"
  OSV_BIN="$bin"
}

echo "==> Checking lockfile is in sync (uv lock --check)"
uv lock --check

ensure_osv_scanner

echo
echo "==> Scanning uv.lock against OSV (osv-scanner v${VERSION})"
# osv-scanner exits non-zero when vulns are found; surface without failing (advisory).
"$OSV_BIN" scan --lockfile=uv.lock || true

echo
echo "Audit complete. Findings above are advisory; cross-check relevance to actual usage."
