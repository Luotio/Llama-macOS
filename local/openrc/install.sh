#!/usr/bin/env bash
# Install or update the Strata OpenRC service on this machine.
#
#   local/openrc/install.sh               install/update init script + conf.d (+ logrotate)
#   local/openrc/install.sh --enable      ... and add the service to the default runlevel
#   local/openrc/install.sh --restart     ... and restart the service if it is running
#   local/openrc/install.sh --force-conf  overwrite an existing /etc/conf.d/strata
#   local/openrc/install.sh --uninstall   stop, remove from runlevels, delete the init
#                                         script (add --purge to delete conf.d and logrotate too)
#
# Re-running is the update path: the init script is always replaced; /etc/conf.d/strata
# is only created when missing (your model choice survives), and a diff against the
# repo template is shown when it differs. The service is never restarted unless
# --restart is given, so an update cannot interrupt a running agent session.
#
# DESTDIR=/some/dir installs into a staging root without root privileges (testing).
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SRC/../.." && pwd)"
DESTDIR="${DESTDIR:-}"
INITD="$DESTDIR/etc/init.d/strata"
CONFD="$DESTDIR/etc/conf.d/strata"
LOGROTATE="$DESTDIR/etc/logrotate.d/strata"

ENABLE=0 RESTART=0 FORCE_CONF=0 UNINSTALL=0 PURGE=0
for arg in "$@"; do
    case "$arg" in
        --enable) ENABLE=1 ;;
        --restart) RESTART=1 ;;
        --force-conf) FORCE_CONF=1 ;;
        --uninstall) UNINSTALL=1 ;;
        --purge) PURGE=1 ;;
        -h | --help) sed -n '2,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $arg (see --help)" >&2; exit 1 ;;
    esac
done

if [[ -z "$DESTDIR" && $EUID -ne 0 ]]; then
    echo "[openrc] root is needed, re-running with sudo"
    exec sudo "$0" "$@"
fi

if [[ -z "$DESTDIR" ]] && ! command -v rc-service >/dev/null; then
    echo "error: OpenRC (rc-service) not found" >&2
    exit 1
fi

live() { [[ -z "$DESTDIR" ]]; }   # real system, not a staging root

if [[ $UNINSTALL -eq 1 ]]; then
    if live; then
        rc-service strata stop 2>/dev/null || true
        rc-update del strata 2>/dev/null || true
    fi
    rm -f "$INITD"
    if [[ $PURGE -eq 1 ]]; then
        rm -f "$CONFD" "$LOGROTATE"
    fi
    echo "[openrc] removed $INITD$([[ $PURGE -eq 1 ]] && echo ", $CONFD, $LOGROTATE")"
    exit 0
fi

mkdir -p "$(dirname "$INITD")" "$(dirname "$CONFD")"

install -m 0755 "$SRC/strata.initd" "$INITD"
echo "[openrc] installed $INITD"

TMP_CONF="$(mktemp)"
trap 'rm -f "$TMP_CONF"' EXIT
sed "s|@REPO@|$REPO_ROOT|g" "$SRC/strata.confd" > "$TMP_CONF"
if [[ ! -e "$CONFD" || $FORCE_CONF -eq 1 ]]; then
    install -m 0644 "$TMP_CONF" "$CONFD"
    echo "[openrc] wrote $CONFD"
elif ! diff -q "$TMP_CONF" "$CONFD" >/dev/null; then
    echo "[openrc] kept your $CONFD (differs from the repo template; --force-conf overwrites):"
    diff -u "$CONFD" "$TMP_CONF" || true
else
    echo "[openrc] $CONFD is up to date"
fi

# The server's console output goes to /var/log/strata/strata.log
if [[ -d "$DESTDIR/etc/logrotate.d" ]]; then
    cat > "$LOGROTATE" <<'EOF'
/var/log/strata/strata.log {
    weekly
    rotate 4
    compress
    missingok
    notifempty
    copytruncate
}
EOF
    chmod 0644 "$LOGROTATE"
    echo "[openrc] installed $LOGROTATE"
fi

if live; then
    if [[ $ENABLE -eq 1 ]]; then
        rc-update add strata default
    fi
    if rc-service strata status >/dev/null 2>&1; then
        if [[ $RESTART -eq 1 ]]; then
            rc-service strata restart
        else
            echo "[openrc] strata is running; run 'rc-service strata restart' (or --restart) to apply changes"
        fi
    else
        echo "[openrc] start it with: rc-service strata start   (stop any manually started server first)"
    fi
fi
