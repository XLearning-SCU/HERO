#!/usr/bin/env bash
# Bound repository/download waits without interrupting dpkg during installation.
# Run in a subshell so temporary files and traps stay local to this invocation.
hero_apt_install() (
  set -euo pipefail
  local script_dir update_timeout download_timeout source_dir source_file rc=1
  script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  update_timeout="${HERO_APT_UPDATE_TIMEOUT_SEC:-180}"
  download_timeout="${HERO_APT_DOWNLOAD_TIMEOUT_SEC:-600}"
  if [[ ! "${update_timeout}" =~ ^[1-9][0-9]*$ || ! "${download_timeout}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[apt] update/download timeouts must be positive integer seconds" >&2
    return 2
  fi
  command -v timeout >/dev/null || { echo "[apt] GNU timeout is required" >&2; return 1; }
  source_dir="$(mktemp -d /tmp/hero-apt-sources.XXXXXX)"
  trap 'rm -rf -- "${source_dir}"' EXIT
  # apt's unprivileged downloader must be able to read these source lists.
  chmod 755 "${source_dir}"
  local -a sources=() options=(
    -o Acquire::Retries=2
    -o Acquire::http::Timeout=30
    -o Acquire::https::Timeout=30
    -o DPkg::Lock::Timeout=120
    -o APT::Update::Error-Mode=any
  )
  if [[ "${HERO_APT_OFFICIAL_SOURCE:-0}" != "1" ]]; then
    sources+=(configured)
  fi
  # Keep the legacy OFFICIAL_SOURCE flag as the switch to skip image sources.
  # Prefer Aliyun for downloads; all fallback repositories remain signed.
  sources+=(https://mirrors.aliyun.com/ubuntu official https://mirrors.cloud.tencent.com/ubuntu https://mirrors.huaweicloud.com/ubuntu)
  local source attempt=0
  for source in "${sources[@]}"; do
    attempt=$((attempt + 1))
    local -a source_options=()
    if [[ "${source}" != configured ]]; then
      source_file="${source_dir}/sources.list"
      if [[ "${source}" == official ]]; then
        cp "${script_dir}/ubuntu-noble.list" "${source_file}"
      else
        printf 'deb %s %s main restricted universe multiverse\n' \
          "${source}" noble "${source}" noble-updates "${source}" noble-security > "${source_file}"
      fi
      chmod 644 "${source_file}"
      source_options=(-o "Dir::Etc::sourcelist=${source_file}" -o Dir::Etc::sourceparts=-)
    fi
    echo "[apt] source ${attempt}/${#sources[@]}=${source}; packages=$*"
    if timeout --signal=TERM --kill-after=15s "${update_timeout}s" \
        apt-get "${options[@]}" "${source_options[@]}" update; then
      :
    else
      rc=$?
      echo "[apt] update failed (exit=${rc}); trying next source" >&2
      continue
    fi
    # Only acquisition is timed out. No package unpack/configuration occurs here.
    if DEBIAN_FRONTEND=noninteractive timeout --signal=TERM --kill-after=15s "${download_timeout}s" \
        apt-get "${options[@]}" "${source_options[@]}" install --download-only -y "$@"; then
      :
    else
      rc=$?
      echo "[apt] download failed (exit=${rc}); trying next source" >&2
      continue
    fi
    # Packages are now local. Preserve real installation errors, and never kill
    # dpkg on a network deadline or repeat its failed configuration on each mirror.
    if DEBIAN_FRONTEND=noninteractive apt-get "${options[@]}" "${source_options[@]}" \
        install --no-download -y "$@"; then
      return 0
    else
      rc=$?
      echo "[apt] local installation failed (exit=${rc})" >&2
      return "${rc}"
    fi
  done
  echo "[apt] all ${#sources[@]} sources failed; required packages were not installed" >&2
  return "${rc}"
)
