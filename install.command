#!/bin/sh
sh "$(dirname "$0")/install.sh" "$@"
result=$?
if [ "$#" -eq 0 ] && [ -t 0 ]; then
    printf 'Press Return to close: '
    read -r reply
fi
exit "$result"
