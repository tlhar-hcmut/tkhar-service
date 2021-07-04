DIR_ENV="$PWD/.venv"


build() {
    local path_success=".ci/success_env.txt"
    [ ! -f "$path_success" ] && rm -rf "$DIR_ENV"
    python3 -m venv "$DIR_ENV"
    touch "$path_success"
    mkdir -p "src"
    pip3 install -e . -i "$PYPI"
    [ "$?" -ne 0 ] && return 1
    if [ -d "tests" ]; then
        pytest "tests"
        local code="$?"
        [ "$code" -ne 5 ] && [ "$code" -ne 0 ] && return 1
    fi
    return 0
}

start() {
    python3 -m "app.main"
}

ACTION="${1?}"
LS_ACTION="build start"
for ACTION in "$@"; do
    [ "$?" -ne 0 ] && exit 1
    mkdir -p ".ci"
    mkdir -p "log"
    "$ACTION"
    [ "$?" -ne 0 ] && exit 1
done
exit 0