#!/usr/bin/env bash
# Install (or update) the llama.cpp monitor OpenRC service, add it to the
# default runlevel (so it starts at boot), and start it.
#
#   sudo local/openrc/llama-mon-install.sh             install + enable + start
#   sudo local/openrc/llama-mon-install.sh --restart   restart a running service
#   sudo local/openrc/llama-mon-install.sh --force-conf  overwrite conf.d + JSON config
#   sudo local/openrc/llama-mon-install.sh --uninstall stop, remove runlevel entry
#
# Re-running is the update path: the init script is always replaced;
# /etc/conf.d/llama-mon and /etc/llama-mon/config.json (the monitor's main
# JSON config, schema: local/monitor/README.md) are only created when
# missing, and diffs against the repo templates are shown when they differ.
#
# DESTDIR=/some/dir stages into a directory instead of the live system (testing).
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SRC/../.." && pwd)"
DESTDIR="${DESTDIR:-}"
INITD="$DESTDIR/etc/init.d/llama-mon"
CONFD="$DESTDIR/etc/conf.d/llama-mon"
JSONC="$DESTDIR/etc/llama-mon/config.json"
CONF_TEMPLATE="$REPO_ROOT/local/monitor/config.example.json"

RESTART=0 FORCE_CONF=0 UNINSTALL=0
for arg in "$@"; do
    case "$arg" in
        --restart) RESTART=1 ;;
        --force-conf) FORCE_CONF=1 ;;
        --uninstall) UNINSTALL=1 ;;
        -h | --help) sed -n '2,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
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
        rc-service llama-mon stop 2>/dev/null || true
        rc-update del llama-mon 2>/dev/null || true
    fi
    rm -f "$INITD"
    echo "[llama-mon] removed $INITD ($CONFD and $JSONC kept; delete them manually if wanted)"
    exit 0
fi

if live && ! command -v rc-service >/dev/null; then
    echo "error: OpenRC (rc-service) not found" >&2
    exit 1
fi

mkdir -p "$(dirname "$INITD")" "$(dirname "$CONFD")"

install -m 0755 "$SRC/llama-mon.initd" "$INITD"
echo "[llama-mon] installed $INITD"

TMP_CONF="$(mktemp)"
trap 'rm -f "$TMP_CONF"' EXIT
sed "s|@REPO@|$REPO_ROOT|g" "$SRC/llama-mon.confd" > "$TMP_CONF"
if [[ ! -e "$CONFD" || $FORCE_CONF -eq 1 ]]; then
    install -m 0644 "$TMP_CONF" "$CONFD"
    echo "[llama-mon] wrote $CONFD"
elif ! diff -q "$TMP_CONF" "$CONFD" >/dev/null; then
    echo "[llama-mon] kept your $CONFD (differs from the repo template; --force-conf overwrites):"
    diff -u "$CONFD" "$TMP_CONF" || true
else
    echo "[llama-mon] $CONFD is up to date"
fi

# The monitor's main JSON config (GPU visibility, price sources, throttle).
mkdir -p "$(dirname "$JSONC")"
if [[ ! -e "$JSONC" || $FORCE_CONF -eq 1 ]]; then
    install -m 0644 "$CONF_TEMPLATE" "$JSONC"
    echo "[llama-mon] wrote $JSONC (edit it, then: rc-service llama-mon restart)"
elif ! diff -q "$CONF_TEMPLATE" "$JSONC" >/dev/null; then
    echo "[llama-mon] kept your $JSONC (differs from the repo template; --force-conf overwrites):"
    diff -u "$JSONC" "$CONF_TEMPLATE" || true
else
    echo "[llama-mon] $JSONC is up to date"
fi

if live; then
    rc-update add llama-mon default
    if rc-service llama-mon status >/dev/null 2>&1; then
        if [[ $RESTART -eq 1 ]]; then
            rc-service llama-mon restart
        else
            echo "[llama-mon] already running; run 'rc-service llama-mon restart' (or --restart) to apply changes"
        fi
    else
        rc-service llama-mon start
        host=$(sed -n 's/^MON_HOST="\(.*\)"/\1/p' "$CONFD" | head -n1)
        port=$(sed -n 's/^MON_PORT="\(.*\)"/\1/p' "$CONFD" | head -n1)
        echo "[llama-mon] started and enabled at boot; dashboard at http://${host:-127.0.0.1}:${port:-8085}/"
    fi
fi
