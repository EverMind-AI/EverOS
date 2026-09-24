# Running EverOS on Windows

> Also available in Chinese: [windows.zh.md](windows.zh.md)

EverOS runs on Windows in two ways. **Natively** — `pip install everos`
on a stock Windows 11 machine, verified end to end and covered by the
`unit tests (Windows)` CI job; that path is [Native Windows](#native-windows)
and is where to start. Or under **WSL2** — a real Linux kernel, so the
storage stack (file locking, LanceDB, inotify) behaves exactly as it does
on a Linux server; the rest of this page is that install, including the
parts that cannot be scripted and the one failure that is silent.

## Table of contents

- [Before you start](#before-you-start)
  - [Will this machine need a reboot?](#will-this-machine-need-a-reboot)
- [Install](#install)
  - [Phase 1 — WSL2 and the distro](#phase-1--wsl2-and-the-distro)
  - [Phase 2 — EverOS inside the distro](#phase-2--everos-inside-the-distro)
  - [Scripted install](#scripted-install)
- [Verify](#verify)
- [Where the memory root must live](#where-the-memory-root-must-live)
- [Office document support](#office-document-support)
- [Reaching the API from Windows](#reaching-the-api-from-windows)
- [Troubleshooting](#troubleshooting)
- [Native Windows](#native-windows)

## Before you start

| Requirement | Notes |
|---|---|
| Windows 11, or Windows 10 build 19041+ | `winver` to check |
| Administrator rights | Enabling the WSL feature needs elevation |
| Hardware virtualization enabled in firmware | Task Manager → Performance → CPU → "Virtualization: Enabled" |
| ~3 GB disk for the distro, ~1 GB more with LibreOffice | |

On a managed corporate machine, Hyper-V and the virtual machine platform
are sometimes blocked by policy. That is a hard stop — it needs IT, not a
workaround.

### Will this machine need a reboot?

`wsl --install` enables the `VirtualMachinePlatform` Windows feature, and
enabling it requires a restart. But the feature is often **already on** —
Docker Desktop, Hyper-V, Windows Sandbox, the Android emulator and
virtualization-based security all turn it on. Check before you plan around
a reboot:

```powershell
(Get-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform).State
```

- `Enabled` → no reboot; the whole install runs in one pass.
- `Disabled` → one reboot, once, on this machine only.

## Install

### Phase 1 — WSL2 and the distro

Run PowerShell **as Administrator**:

```powershell
wsl --install --no-launch
```

`--no-launch` matters: without it, `wsl --install` opens the distro's
first-run wizard and blocks on an interactive "Enter new UNIX username"
prompt, which is what breaks unattended installs.

Reboot if the preflight above said `Disabled`. Then create the distro
without the interactive account setup:

```powershell
wsl --install -d Ubuntu --no-launch
ubuntu install --root
```

### Phase 2 — EverOS inside the distro

```bash
wsl -d Ubuntu -u root -- bash -lc '
  set -euo pipefail
  apt-get update -qq
  apt-get install -y -qq python3-venv curl
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  uv venv --python 3.12 /opt/everos
  uv pip install --python /opt/everos/bin/python everos
'
```

### Scripted install

The two phases combine into one idempotent script. Re-running it after the
reboot picks up where it stopped.

```powershell
# everos-setup.ps1 — run as Administrator
$ErrorActionPreference = 'Stop'

# Phase 1: WSL platform.
if (-not (Get-Command wsl -ErrorAction SilentlyContinue) -or -not (wsl --version 2>$null)) {
    wsl --install --no-launch
    Write-Host 'WSL installed. Reboot, then re-run this script.' -ForegroundColor Yellow
    exit 0
}

# Phase 2: distro.
if (-not (wsl -l -q | Select-String -Quiet 'Ubuntu')) {
    wsl --install -d Ubuntu --no-launch
    ubuntu install --root
}

# Phase 3: EverOS.
wsl -d Ubuntu -u root -- bash -lc @'
set -euo pipefail
apt-get update -qq
apt-get install -y -qq python3-venv curl
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.12 /opt/everos
uv pip install --python /opt/everos/bin/python everos
'@

Write-Host 'Done. Start with: wsl -d Ubuntu -- /opt/everos/bin/everos serve' -ForegroundColor Green
```

Detect WSL with `wsl --version`, not `wsl --status` — the latter's exit
code is not a reliable "is it installed" signal.

## Verify

```bash
wsl -d Ubuntu -- /opt/everos/bin/everos --version
wsl -d Ubuntu -- /opt/everos/bin/everos init
wsl -d Ubuntu -- /opt/everos/bin/everos serve
```

From there follow the [Quick Start](../README.md#quick-start) — inside the
distro everything behaves as on any Ubuntu host.

## Where the memory root must live

> [!IMPORTANT]
> Keep the memory root on the WSL2 filesystem (`/home/...`, `/opt/...`).
> **Never put it under `/mnt/c/`.**

Filesystem events do not propagate from the Windows host into WSL2. If the
memory root sits on a `/mnt/c` mount, the cascade watcher starts without
error, logs normally, and receives **zero events** — edits to markdown
files never reach the index, and searches silently answer from stale data.
Nothing in the logs says so.

If you must keep files on the Windows side, fall back to polling:

| Option | How |
|---|---|
| Scanner sweep (default 30 s) | Already on; bounded but eventually consistent |
| Faster sweep | Drop the scan interval to ~5 s for a small root |
| Explicit sync | `everos cascade sync` after batch edits |

Details in the [cascade runbook](cascade_runbook.md#wsl2--network-mounts).

## Office document support

Install the **Linux** LibreOffice inside the distro — the Windows build
cannot serve a process running in WSL2:

```bash
wsl -d Ubuntu -u root -- apt-get install -y libreoffice
```

The parser resolves the binary with `shutil.which("soffice")`, a plain
PATH lookup inside the distro, so a Windows `soffice.exe` is never found.
Without it, office uploads return `503 CAPABILITY_UNAVAILABLE`; see
[multimodal.md](multimodal.md#libreoffice-office-documents-only).

## Reaching the API from Windows

Yes — WSL2 forwards `localhost` by default (`localhostForwarding`), so a
server started inside the distro answers on the same port from Windows.
`everos serve` binds `127.0.0.1` inside the distro; the forwarder relays
to it, so the loopback default does not need changing:

```bash
wsl -d Ubuntu -- /opt/everos/bin/everos serve   # listens inside the distro
curl http://localhost:8000/health               # from Windows PowerShell
```

If that call does not connect, the usual causes are a VPN client taking
over the WSL network, or a wedged forwarder. Restart WSL first:

```powershell
wsl --shutdown        # next `wsl` command restarts it and rebuilds the relay
```

> [!WARNING]
> Do not reach for `host = "0.0.0.0"` as a fix. **EverOS ships no
> authentication of its own** — loopback is what keeps the API private
> (see [SECURITY.md](../SECURITY.md)). How far `0.0.0.0` actually reaches
> depends on the WSL networking mode, and the two differ sharply:
>
> | `networkingMode` in `.wslconfig` | What `0.0.0.0` exposes |
> |---|---|
> | `nat` (default) | The distro's private virtual network plus the Windows host. Other machines on your LAN cannot reach it without an explicit `netsh portproxy`. |
> | `mirrored` (WSL 2.0+) | Every interface the Windows machine has — **the API becomes reachable from the LAN**. |
>
> Only bind `0.0.0.0` once your own gateway or auth layer sits in front.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `wsl --install` hangs on "Enter new UNIX username" | `--no-launch` omitted | Ctrl-C, re-run with `--no-launch`, then `ubuntu install --root` |
| "The virtual machine could not be started" | Virtualization off in firmware | Enable VT-x / AMD-V in BIOS/UEFI |
| `WslRegisterDistribution failed with error: 0x80370102` | Same, or Hyper-V blocked by policy | Firmware setting, or IT |
| Install succeeded but nothing runs after reboot | Script not re-run | Re-run it; it is idempotent |
| Markdown edits never appear in search | Memory root on `/mnt/c` | Move it into the distro filesystem |
| Office upload returns `503` | LibreOffice missing **inside** the distro | `apt-get install -y libreoffice` |

## Native Windows

EverOS runs directly on Windows. Verified on a stock Windows 11 Enterprise
laptop (Intel Core Ultra 7 155H, 32 GB, no Visual C++ Redistributable
installed) with Python 3.12 from `uv`:

- `pip install` of the built wheel into a plain venv (dependencies from
  PyPI: pyarrow 25.0.1, `msvc-runtime` 14.44, lancedb 0.34) →
  `everos init` → put an LLM `api_key` in `everos.toml` (the server refuses
  to start without one, and `init` says so) → `everos server start`: healthy
  after 35 s, `/add` and `/search` answered. No other manual step. The
  Windows-only dependencies are `msvc-runtime`, which supplies the C++
  runtime `greenlet` needs (details below), and `pywin32`, which
  `portalocker` uses for file locking.
- Test suites: the unit suite runs green in CI on `windows-latest`
  (`unit tests (Windows)`, 2583 passed / 4 skipped — the same count as
  Linux); on that machine, integration **183 passed / 5 skipped** and live
  LLM (`slow`) **28 passed / 1 skipped**, both under Python 3.12.
- All four memory kinds — episode, profile, agent case, agent skill — were
  produced by a Tier 3 server (real LLM, embedding and rerank providers) and
  read back through `/get` and `/search`.
- A 10-hour soak (about 78 000 markdown entries written and rewritten,
  16 000 searches, 2 300 `/add` calls through the extraction path, two
  concurrent `everos cascade sync` processes on the same tree) ended with
  the index intact: every table opens, schemas verify, and every well-formed
  entry had its row — the only markdown entries without one were the
  deliberately malformed files the run seeds. RSS levelled at about 2.3 GB
  after three hours; the LanceDB directory peaked at 6.7 GB and reclaimed
  to 2.7 GB (437 MB of live data). Two findings from that run are tracked
  separately and are not Windows-specific: concurrent `cascade sync`
  processes can insert the same row twice (about 4.5 % duplicate rows after
  10 hours, no corruption), and request latency degrades under sustained
  write load.

Python: 3.12 is what ran on that machine; 3.13 and 3.14 (regular build)
pass the same suites in CI and on macOS without changes; 3.11 is refused by
`requires-python` and by the PEP 695 syntax in `src/`; the free-threaded
3.14t build has no `lancedb` wheel.

Windows specifics worth knowing:

- `os.replace` fails with `PermissionError` while another process — the
  cascade worker, or Defender scanning a fresh file — holds the target open.
  The markdown writer retries with backoff (about 2.5 s of patience) and
  logs each retry at debug level; the soak's load generator, which uses
  the same idiom, hit 49 such violations and recovered from all of them.
- Windows Search indexes everything under `%USERPROFILE%`. A memory root
  there costs about one CPU core of `SearchIndexer` under heavy writes; put
  the root elsewhere or exclude the directory from indexing.
- File-change events come from `ReadDirectoryChangesW`, which can report a
  rename as two events. The cascade scanner reconciles against the disk.
- Both machines that ran the suites had long paths enabled
  (`HKLM\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled=1`;
  GitHub's image sets it, a stock install does not). Memory roots nest
  several directories deep and Lance index directories carry UUID names,
  so a root under a long profile path can cross 260 characters — enable
  long paths or keep the root short.

The one prerequisite that used to bite on a clean machine is the
**Microsoft Visual C++ Redistributable (x64)**. `greenlet`, which
SQLAlchemy's async engine depends on, is a C++ extension whose wheel does
not bundle the runtime, so without it every SQLite call fails with a cryptic
`DLL load failed while importing _greenlet`. GitHub's CI image has the
redistributable preinstalled, which is why CI never caught this. EverOS now
carries the runtime itself: the Windows-only `msvc-runtime` dependency puts
the DLLs in `sys.prefix`, and `everos/__init__.py` registers that directory
with the DLL loader before anything else is imported. Nothing to install,
no administrator rights needed. One dependency to watch: `msvc-runtime`
ships wheels only, one per CPython minor version, so a Python newer than
its latest wheel cannot install EverOS on Windows until upstream publishes
one.
