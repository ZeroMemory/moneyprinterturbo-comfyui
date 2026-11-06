#!/bin/sh

target=${MPT_ADDON_BOOTSTRAP_TARGET:-}
requested_python=
next=
for arg do
    if [ "$next" = python ]; then requested_python=$arg; next=; continue; fi
    if [ "$next" = skip ]; then next=; continue; fi
    case "$arg" in
        --mpt-python) next=python ;;
        --mpt-python=*) requested_python=${arg#*=} ;;
        install|configure|uninstall|--help|-h|--external-server|--allow-insecure-comfyui-http) ;;
        --*=*) ;;
        --*|-*) next=skip ;;
        *) if [ -z "$target" ]; then target=$arg; fi ;;
    esac
done

valid_python() {
    "$1" -c 'import sys; raise SystemExit(sys.version_info < (3, 11))' >/dev/null 2>&1
}

project_python() {
    [ -n "$target" ] || return
    case "$target" in
        '~') target=$HOME ;;
        '~/'*) target="$HOME/${target#\~/}" ;;
        \"*\") target=${target#\"}; target=${target%\"} ;;
    esac
    for candidate in "$target/.venv/bin/python" "$target/lib/python/python" "$target/lib/python/bin/python" "$target/bin/python"; do
        if [ -x "$candidate" ] && valid_python "$candidate"; then
            starter=$candidate
            return
        fi
    done
}

starter=
if [ -n "$requested_python" ]; then
    if ! valid_python "$requested_python"; then
        printf '%s\n' 'The --mpt-python executable must run Python 3.11 or later.' >&2
        exit 1
    fi
    starter=$requested_python
else
    project_python
    if [ -z "$starter" ]; then
        for candidate in python3 python; do
            if valid_python "$candidate"; then starter=$candidate; break; fi
        done
    fi
fi

if [ -z "$starter" ] && [ "$#" -eq 0 ] && [ -t 0 ]; then
    printf 'MoneyPrinterTurbo folder: '
    IFS= read -r target || exit 1
    project_python
    if [ -n "$starter" ]; then export MPT_ADDON_BOOTSTRAP_TARGET="$target"; fi
fi
if [ -z "$starter" ]; then
    printf '%s\n' 'Python 3.11 or later was not found. Pass the MPT folder or use --mpt-python with its Python executable.' >&2
    exit 1
fi
exec "$starter" "$(dirname "$0")/install.py" "$@"
