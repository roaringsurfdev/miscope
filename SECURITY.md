# Security

MIScope is a research instrument for analyzing how transformer models change
during training. This document states what the project does to stay safe for the
people who use it, and — just as importantly — where that responsibility ends.

## Statement of intent

Security hygiene is treated as ongoing maintenance, not a one-time checkbox,
within the reasonable scope of a research library:

- Dependencies are pinned and integrity-checked, and audited on a schedule (see
  *Dependency hygiene* below).
- Model and artifact loading use formats that do **not** execute code on load
  (see *How MIScope loads data* below).
- Security-relevant choices in the code are made deliberately and documented, so
  a reviewer can verify the posture rather than trust a claim.

This is a best-effort posture appropriate to a research tool. It is not a
warranty (see the license).

## How MIScope loads data

The two ways MIScope reads data from disk are both designed to avoid the most
common code-execution vector in the ML ecosystem — deserializing untrusted
pickled objects (the risk in `torch.load`'s default and in `pickle`):

- **Model checkpoints use safetensors.** Checkpoints are read with
  `safetensors.torch.load_file` (`Variant.load_checkpoint`). The safetensors
  format stores raw tensor data with a JSON header and contains no executable
  code, so loading a checkpoint cannot run arbitrary code the way a pickle-based
  format can. The library contains no `torch.load`, `pickle`, or
  auto-downloading (`from_pretrained` / Hugging Face Hub) code paths.
- **Analysis artifacts use NumPy `.npz` with `allow_pickle=False`.** Artifacts
  produced by the analysis pipeline are read without object deserialization
  (the NumPy 2.x default, and set explicitly on the cross-epoch path).

## Scope and the user's responsibility

MIScope analyzes checkpoints produced by its own training scripts, read from a
local data root that **you** control. The safeguards above mean that *loading* a
checkpoint does not execute code embedded in the file. They do **not** make it
safe to run untrusted code or to trust the contents of arbitrary third-party
artifacts in other respects:

- If you point MIScope at checkpoints or data obtained from an untrusted source,
  the safetensors/`.npz` formats still prevent code execution on load, but the
  project cannot vouch for the correctness, provenance, or research validity of
  external artifacts.
- The project is not responsible for the security of the broader environment in
  which it runs (your OS, your other tools, model files you load through other
  libraries outside MIScope).
- If you extend MIScope with code that loads checkpoints via pickle-based formats
  (`torch.load`, `pickle`, `joblib`), you reintroduce that risk yourself; prefer
  safetensors.

In short: MIScope aims to be safe to *run on your own training outputs*. It is
not a sandbox for executing or validating untrusted model files.

## Dependency hygiene

- Dependencies are locked with hashes (`uv.lock`); installs fail on a hash
  mismatch.
- The lock is scanned against the OSV vulnerability database both on a weekly
  schedule and on every pull request (`.github/workflows/audit.yml`), with an
  equivalent local script (`scripts/audit_deps.sh`).
- The scanner itself is pinned to a specific version and verified by checksum
  (`.github/osv-scanner.pin`); a tampered or substituted download fails closed.

Findings are advisory and triaged by relevance to actual usage. As a local
research tool, MIScope's exposure to many web-stack advisories is limited;
deserialization issues in the ML stack are the category watched most closely.

## Reporting a vulnerability

If you find a security issue, please report it privately rather than opening a
public issue. <!-- TODO: confirm a contact (private email or GitHub Security
Advisories) before publishing this file. --> Include steps to reproduce and the
affected version or commit. As a small research project, response is best-effort.
