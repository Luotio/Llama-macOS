#!/usr/bin/env bash
# Install (or update) the llama.cpp monitor as a *systemd* service. The code
# is COPIED to an install prefix (/opt/llama-mon by default, override with
# LLAMA_MON_PREFIX=/path) and the service runs from there, never out of the
# git checkout: no repo-path coupling, no half-checked-out state serving the
# dashboard, no config conflicts between repo templates and live files.
# Live configuration stays in /etc/llama-mon/config.json.
#
# Updating = re-run this script, then restart (--restart does both).
#
#   sudo local/systemd/install.sh                    install + enable + start
#   sudo local/systemd/install.sh --restart          refresh code + restart
#   sudo local/systemd/install.sh --no-start         install/enable but do not start
#   sudo local/systemd/install.sh --force-conf       overwrite /etc/llama-mon/config.json
#   sudo local/systemd/install.sh --with-sudoers=USER  let service user USER cap GPUs via sudo -n
#                                                    (also set User=USER in the unit)
#   sudo local/systemd/install.sh --uninstall        stop, disable, remove unit + code
#
# Re-running is the update path: the unit and the copied code are always
# replaced; /etc/llama-mon/config.json is only created when missing, and a
# diff against the repo template is shown when it differs. If the OpenRC
# llama-mon exists, this script refuses to run both monitors at once (same
# port): uninstall that one first.
#
# DESTDIR=/some/dir stages into a directory instead of the live system (testing).
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SRC/../.." && pwd)"
DESTDIR="${DESTDIR:-}"
PREFIX_LIVE="/opt/llama-mon"                      # where the service runs from
PREFIX="${DESTDIR}${LLAMA_MON_PREFIX:-$PREFIX_LIVE}"
UNIT="$DESTDIR/etc/systemd/system/llama-mon.service"
CONF_DIR="$DESTDIR/etc/llama-mon"
CONF="$CONF_DIR/config.json"
CONF_TEMPLATE="$REPO_ROOT/local/monitor/config.example.json"
MON_SRC="$REPO_ROOT/local/monitor/monitor.py"
README_SRC="$REPO_ROOT/local/monitor/README.md"
SUDOERS="$DESTDIR/etc/sudoers.d/llama-mon"

RESTART=0 FORCE_CONF=0 UNINSTALL=0 NO_START=0 SUDOERS_USER=""
for arg in "$@"; do
    case "$arg" in
        --restart) RESTART=1 ;;
        --force-conf) FORCE_CONF=1 ;;
        --uninstall) UNINSTALL=1 ;;
        --no-start) NO_START=1 ;;
        --with-sudoers=*) SUDOERS_USER="${arg#*=}" ;;
        -h | --help) sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $arg (see --help)" >&2; exit 1 ;;
    esac
done

if [[ -z "$DESTDIR" && $EUID -ne 0 ]]; then
    echo "[llama-mon] root is needed, re-running with sudo"
    exec sudo "$0" "$@"
fi

live() { [[ -z "$DESTDIR" ]]; }

if [[ $UNINSTALL -eq 1 ]]; then
    if live; then
        systemctl stop llama-mon.service 2>/dev/null || true
        systemctl disable llama-mon.service 2>/dev/null || true
    fi
    rm -f "$UNIT" "$SUDOERS"
    rm -rf "$PREFIX"
    [[ -d "$DESTDIR/etc/systemd/system" ]] && systemctl daemon-reload 2>/dev/null || true
    echo "[llama-mon] removed $UNIT and $PREFIX ($CONF kept; delete it manually if unwanted)"
    exit 0
fi

if live; then
    if ! command -v systemctl >/dev/null; then
        echo "error: systemd (systemctl) not found" >&2
        exit 1
    fi
    if [[ -e /etc/init.d/llama-mon ]] && rc-service llama-mon status >/dev/null 2>&1; then
        echo "error: the OpenRC llama-mon is installed and running; both would fight over" >&2
        echo "       the same dashboard port. stop+disable it first:" >&2
        echo "         sudo rc-service llama-mon stop && sudo rc-update del llama-mon" >&2
        echo "       (or remove it: sudo local/openrc/llama-mon-install.sh --uninstall)" >&2
        exit 1
    fi
fi

[[ -f "$MON_SRC" ]] || { echo "error: $MON_SRC missing" >&2; exit 1; }

# --- copy the code into the install prefix ---------------------------------
mkdir -p "$PREFIX"
install -m 0644 "$MON_SRC" "$PREFIX/monitor.py"
install -m 0644 "$CONF_TEMPLATE" "$PREFIX/config.example.json"
[[ -f "$README_SRC" ]] && install -m 0644 "$README_SRC" "$PREFIX/README.md"
echo "[llama-mon] installed code -> ${PREFIX#$DESTDIR}/ (live runs from $PREFIX_LIVE)"

# --- render + install the unit (substitutes @PREFIX@ with the live path) ---
mkdir -p "$(dirname "$UNIT")"
sed "s|@PREFIX@|$PREFIX_LIVE|g" "$SRC/llama-mon.service" > "$UNIT"
chmod 0644 "$UNIT"
echo "[llama-mon] installed $UNIT"

# --- /etc config: seed once, never clobber ----------------------------------
mkdir -p "$CONF_DIR"
if [[ ! -e "$CONF" || $FORCE_CONF -eq 1 ]]; then
    install -m 0644 "$CONF_TEMPLATE" "$CONF"
    echo "[llama-mon] wrote $CONF (edit it, then: systemctl restart llama-mon)"
elif ! diff -q "$CONF_TEMPLATE" "$CONF" >/dev/null; then
    echo "[llama-mon] kept your $CONF (differs from the repo template; --force-conf overwrites):"
    diff -u "$CONF" "$CONF_TEMPLATE" || true
else
    echo "[llama-mon] $CONF is up to date"
fi

if [[ -n "$SUDOERS_USER" ]]; then
    TMP_SUDO="$(mktemp)"
    trap 'rm -f "$TMP_SUDO"' EXIT
    sed "s/^MONUSER/$SUDOERS_USER/" "$SRC/llama-mon.sudoers" > "$TMP_SUDO"
    mkdir -p "$(dirname "$SUDOERS")"
    if live; then
        visudo -c -f "$TMP_SUDO" >/dev/null
        install -m 0440 "$TMP_SUDO" "$SUDOERS"
        echo "[llama-mon] wrote $SUDOERS for user $SUDOERS_USER"
        echo "[llama-mon] note: set 'User=$SUDOERS_USER' in $UNIT for the unprivileged setup"
    else
        install -m 0440 "$TMP_SUDO" "$SUDOERS"
        echo "[llama-mon] staged $SUDOERS"
    fi
fi

if live; then
    systemctl daemon-reload
    systemctl enable llama-mon.service >/dev/null 2>&1 || true
    if systemctl is-active --quiet llama-mon.service; then
        if [[ $RESTART -eq 1 ]]; then
            systemctl restart llama-mon.service
            echo "[llama-mon] restarted (now running the freshly copied code)"
        else
            echo "[llama-mon] already running; run 'systemctl restart llama-mon' (or --restart) to apply changes"
        fi
    elif [[ $NO_START -eq 0 ]]; then
        systemctl start llama-mon.service
        host=$(sed -n 's/.*"host": "\([^"]*\)".*/\1/p' "$CONF" | head -n1)
        port=$(grep -o '"port": *[0-9]*' "$CONF" | head -n1 | grep -o '[0-9]*')
        echo "[llama-mon] started and enabled at boot; dashboard at http://${host:-127.0.0.1}:${port:-8085}/"
    else
        echo "[llama-mon] installed + enabled; start with: systemctl start llama-mon"
    fi
fi
