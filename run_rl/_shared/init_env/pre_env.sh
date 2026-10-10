#!/usr/bin/env bash
# Install host-side prerequisites required before creating the project venv.
# Run this once before install_env.sh. It is safe to run again.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

command -v "${PYTHON_BIN}" >/dev/null || {
  echo "Python 3.12 is required but ${PYTHON_BIN} was not found" >&2
  exit 1
}

python_version="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
[[ "${python_version}" == "3.12" ]] || {
  echo "Python 3.12 is required; found ${python_version}" >&2
  exit 1
}

# Some base images provide the Python uv package without its Rust executable.
# Force-reinstalling installs the executable under ~/.local/bin when needed.
export PATH="${HOME}/.local/bin:${PATH}"
if ! command -v uv >/dev/null || ! uv --version >/dev/null 2>&1; then
  "${PYTHON_BIN}" -m pip install --user --upgrade --force-reinstall uv
fi

command -v uv >/dev/null || {
  echo "uv installation finished but the executable is not on PATH" >&2
  exit 1
}
uv --version

# SWE-Gym executes each task in a local Docker container.  Install only the
# client/daemon package when it is absent; Docker's data root and daemon flags
# are deliberately owned by the project's guard below, rather than /etc/docker.
if ! command -v docker >/dev/null || ! command -v dockerd >/dev/null \
    || ! command -v containerd >/dev/null || ! command -v iptables >/dev/null; then
  command -v apt-get >/dev/null || {
    echo "Docker is required but neither Docker nor apt-get is available" >&2
    exit 1
  }
  source "${SCRIPT_DIR}/apt_install.sh"
  hero_apt_install docker.io containerd iptables
fi
for required_command in docker dockerd containerd iptables; do
  command -v "${required_command}" >/dev/null || {
    echo "Required host command is missing after installation: ${required_command}" >&2
    exit 1
  }
done

source "${PROJECT_ROOT}/run_rl/_shared/runtime/docker_storage_guard.sh"
if [[ "${HERO_REVERT_RUNC_WRAPPER:-0}" == "1" ]]; then
  hero_revert_runc_rootless_wrapper
  echo "Restored the system runc binaries."
  exit 0
fi
export HERO_LOCAL_DOCKER_HOST="${HERO_LOCAL_DOCKER_HOST:-tcp://127.0.0.1:2377}"
export HERO_LOCAL_DOCKER_DATA_ROOT="${HERO_LOCAL_DOCKER_DATA_ROOT:-${XDG_CACHE_HOME:-${HOME}/.cache}/hero/docker-data}"
hero_check_docker_storage

echo "Host prerequisites ready: Python ${python_version}, $(command -v uv), Docker ${HERO_LOCAL_DOCKER_HOST}"
