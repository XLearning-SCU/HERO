#!/usr/bin/env bash
# Dedicated local-Docker preflight and label-scoped cleanup for HERO SWE runs.

hero_docker_host() {
  printf '%s' "${HERO_LOCAL_DOCKER_HOST:-${DOCKER_HOST:-tcp://127.0.0.1:2377}}"
}

# Do not begin an independent train/eval run when a prior workload has already
# consumed the host's RAM.  This is deliberately a launch-time guard only: it
# does not alter model, rollout, eval, CPU, GPU, or Docker concurrency knobs.
# Set HERO_MIN_AVAILABLE_MEMORY_GB=0 to opt out for an intentional stress run.
hero_check_available_memory() {
  local minimum_gb="${HERO_MIN_AVAILABLE_MEMORY_GB:-96}" available_kb available_gb
  [[ "${minimum_gb}" =~ ^[0-9]+$ ]] || {
    echo "[hero] HERO_MIN_AVAILABLE_MEMORY_GB must be a non-negative integer" >&2
    return 2
  }
  (( minimum_gb == 0 )) && return 0
  available_kb="$(awk '/^MemAvailable:/ {print $2; exit}' /proc/meminfo)"
  [[ "${available_kb}" =~ ^[0-9]+$ ]] || {
    echo "[hero] could not determine available host memory" >&2
    return 1
  }
  available_gb=$((available_kb / 1024 / 1024))
  echo "[hero] available host memory=${available_gb} GiB (launch minimum=${minimum_gb} GiB)"
  (( available_gb >= minimum_gb )) || {
    echo "[hero] refusing to launch: available host memory is below ${minimum_gb} GiB; stop or clean the prior eval/train workload first" >&2
    return 1
  }
}

hero_validate_docker_data_root() {
  local data_root="$1"
  if [[ -z "${data_root}" || "${data_root}" != /* ]]; then
    echo "[hero] refusing invalid Docker data-root: ${data_root:-<empty>}" >&2
    return 1
  fi
  case "${data_root}" in
    /|/root|/home|/tmp|/workspaces|/ring|/ring/share|/ring/share/users)
      echo "[hero] refusing broad Docker data-root: ${data_root}" >&2
      return 1
      ;;
  esac
  if [[ "${data_root}" != "${XDG_CACHE_HOME:-${HOME}/.cache}/hero/docker-data" \
        && "${HERO_LOCAL_DOCKER_ALLOW_CUSTOM_DATA_ROOT:-0}" != "1" ]]; then
    echo "[hero] refusing custom Docker data-root: ${data_root}" >&2
    return 1
  fi
}

# Some DinD hosts expose cgroup v1 controllers read-only.  The compatibility
# wrapper below keeps Docker usable there, but it must be the fallback: it
# deliberately removes OCI resource limits.  hero_prepare_docker_cgroups()
# first tries to make private cgroup mounts writable and only restores stock
# runc when that complete probe is safe.
hero_runc_rootless_wrapper_active() {
  local wrapper="${HERO_RUNC_PATH:-/usr/bin/runc}"
  [[ -f "${wrapper}" ]] || return 1
  grep -aq 'HERO_CGROUP_RO_FALLBACK' "${wrapper}" 2>/dev/null && return 0
  # Recognise wrappers installed by older revisions, before the marker was
  # added, so they can be upgraded or safely reverted.
  grep -aq 'linux\["resources"\] = {}' "${wrapper}" 2>/dev/null \
    && grep -aq '/usr/bin/runc.real' "${wrapper}" 2>/dev/null
}

hero_install_runc_rootless_wrapper() {
  local wrapper="${HERO_RUNC_PATH:-/usr/bin/runc}"
  local real="${HERO_RUNC_REAL_PATH:-/usr/bin/runc.real}"
  local sbin="${HERO_RUNC_SBIN_PATH:-/usr/sbin/runc}"
  local sbin_real="${HERO_RUNC_SBIN_REAL_PATH:-/usr/sbin/runc.real}"
  local temporary="${wrapper}.hero.$$"
  command -v python3 >/dev/null || return 1
  if [[ ! -f "${real}" ]]; then
    [[ -x "${wrapper}" ]] || return 1
    mv "${wrapper}" "${real}"
  fi
  cat >"${temporary}" <<'PYWRAP'
#!/usr/bin/env python3
# HERO_CGROUP_RO_FALLBACK
import json
import os
import sys

VALUE_FLAGS = {
    "--root", "--log", "--log-format", "--bundle", "--pid-file",
    "--console-socket", "--preserve-fds", "--criu", "--rootless-config-path",
}
SUBCOMMANDS = {"create", "run"}
raw_args = sys.argv[1:]
args = [arg for arg in raw_args if arg != "--rootless=true"]
subcommand = None
bundle = None
i = 0
while i < len(args):
    arg = args[i]
    if arg in VALUE_FLAGS:
        if arg == "--bundle" and i + 1 < len(args):
            bundle = args[i + 1]
        i += 2
        continue
    if arg.startswith("--bundle="):
        bundle = arg.split("=", 1)[1]
        i += 1
        continue
    if arg.startswith("-"):
        i += 1
        continue
    if subcommand is None:
        subcommand = arg
    i += 1

if subcommand in SUBCOMMANDS:
    config = os.path.join(bundle or os.getcwd(), "config.json")
    try:
        with open(config) as source:
            data = json.load(source)
        linux = data.get("linux") or {}
        linux["cgroupsPath"] = ""
        linux["resources"] = {}
        if isinstance(linux.get("namespaces"), list):
            linux["namespaces"] = [
                item for item in linux["namespaces"]
                if item.get("type") != "cgroup"
            ]
        data["linux"] = linux
        with open(config, "w") as destination:
            json.dump(data, destination)
    except Exception:
        pass

os.execv("/usr/bin/runc.real", ["/usr/bin/runc.real", "--rootless=true"] + raw_args)
PYWRAP
  chmod +x "${temporary}"
  mv -f "${temporary}" "${wrapper}"

  if [[ ! -e "${sbin_real}" && -e "${sbin}" && ! -L "${sbin}" ]]; then
    mv "${sbin}" "${sbin_real}"
  fi
  ln -sf "${wrapper}" "${sbin}"
  "${wrapper}" --version >/dev/null 2>&1
}

hero_revert_runc_rootless_wrapper() {
  local wrapper="${HERO_RUNC_PATH:-/usr/bin/runc}"
  local real="${HERO_RUNC_REAL_PATH:-/usr/bin/runc.real}"
  local sbin="${HERO_RUNC_SBIN_PATH:-/usr/sbin/runc}"
  local sbin_real="${HERO_RUNC_SBIN_REAL_PATH:-/usr/sbin/runc.real}"
  if ! hero_runc_rootless_wrapper_active; then
    "${wrapper}" --version >/dev/null 2>&1
    return
  fi
  [[ -x "${real}" ]] || return 1
  [[ -L "${sbin}" ]] && rm -f "${sbin}"
  [[ -f "${sbin_real}" ]] && mv -f "${sbin_real}" "${sbin}"
  # mv is atomic on /usr, avoiding a window in which dockerd cannot find runc.
  mv -f "${real}" "${wrapper}"
  "${wrapper}" --version >/dev/null 2>&1
}

# Inventory populated by hero_collect_cgroup_v1_mounts.  Keeping this in Bash
# avoids spawning Python while the machine may already be short of processes.
_HERO_CGROUP_MOUNTS=()
_HERO_CGROUP_SUPER_OPTIONS=()
_HERO_CGROUP_PRIVATE=()
_HERO_CGROUP_REMOUNTED=()
_HERO_CGROUP_CREATED_DIRS=()

hero_collect_cgroup_v1_mounts() {
  local mountinfo="${HERO_PROC_MOUNTINFO:-/proc/self/mountinfo}"
  local cgroup_root="${HERO_CGROUP_ROOT:-/sys/fs/cgroup}"
  local line before after filesystem source super_options mountpoint optional field
  local -a fields=()
  local -A index_by_mount=()
  local index private

  _HERO_CGROUP_MOUNTS=()
  _HERO_CGROUP_SUPER_OPTIONS=()
  _HERO_CGROUP_PRIVATE=()
  [[ -r "${mountinfo}" ]] || return 1
  while IFS= read -r line; do
    [[ "${line}" == *" - cgroup "* ]] || continue
    before="${line%% - *}"
    after="${line#* - }"
    read -r filesystem source super_options _ <<<"${after}"
    [[ "${filesystem}" == "cgroup" ]] || continue
    read -r -a fields <<<"${before}"
    (( ${#fields[@]} >= 6 )) || continue
    printf -v mountpoint '%b' "${fields[4]}"
    case "${mountpoint}" in
      "${cgroup_root}"|"${cgroup_root}"/*) ;;
      *) continue ;;
    esac
    private=1
    for field in "${fields[@]:6}"; do
      case "${field}" in
        shared:*|master:*|propagate_from:*) private=0 ;;
      esac
    done
    if [[ -n "${index_by_mount[${mountpoint}]+set}" ]]; then
      index="${index_by_mount[${mountpoint}]}"
      _HERO_CGROUP_SUPER_OPTIONS[${index}]="${super_options}"
      _HERO_CGROUP_PRIVATE[${index}]="${private}"
    else
      index="${#_HERO_CGROUP_MOUNTS[@]}"
      index_by_mount["${mountpoint}"]="${index}"
      _HERO_CGROUP_MOUNTS+=("${mountpoint}")
      _HERO_CGROUP_SUPER_OPTIONS+=("${super_options}")
      _HERO_CGROUP_PRIVATE+=("${private}")
    fi
  done <"${mountinfo}"
  (( ${#_HERO_CGROUP_MOUNTS[@]} > 0 ))
}

hero_cgroup_super_has_controller() {
  local super_options="$1" controller="$2"
  [[ ",${super_options}," == *",${controller},"* ]]
}

hero_validate_required_cgroup_controllers() {
  local required="${HERO_LOCAL_DOCKER_REQUIRED_CGROUP_CONTROLLERS:-cpu,cpuacct,cpuset,memory,pids,blkio,devices}"
  local controller super_options found
  required="${required//,/ }"
  for controller in ${required}; do
    found=0
    for super_options in "${_HERO_CGROUP_SUPER_OPTIONS[@]}"; do
      if hero_cgroup_super_has_controller "${super_options}" "${controller}"; then
        found=1
        break
      fi
    done
    if (( ! found )); then
      echo "[hero] cgroup v1 controller is unavailable: ${controller}" >&2
      return 1
    fi
  done
}

hero_cgroup_probe_mount() {
  local mountpoint="$1" probe="${1}/.hero-cgroup-probe-$$-${RANDOM:-0}"
  mkdir "${probe}" 2>/dev/null || return 1
  rmdir "${probe}" 2>/dev/null
}

hero_make_cgroup_mount_writable() {
  local mountpoint="$1"
  hero_cgroup_probe_mount "${mountpoint}" && return 0
  mount --bind "${mountpoint}" "${mountpoint}" >/dev/null 2>&1 || return 1
  if ! mount -o remount,rw,bind "${mountpoint}" >/dev/null 2>&1 \
      || ! hero_cgroup_probe_mount "${mountpoint}"; then
    umount "${mountpoint}" >/dev/null 2>&1 || true
    return 1
  fi
  _HERO_CGROUP_REMOUNTED+=("${mountpoint}")
}

hero_rollback_cgroup_remounts() {
  local index
  for ((index=${#_HERO_CGROUP_REMOUNTED[@]} - 1; index >= 0; index--)); do
    umount "${_HERO_CGROUP_REMOUNTED[${index}]}" >/dev/null 2>&1 || true
  done
  _HERO_CGROUP_REMOUNTED=()
}

hero_active_swe_container_ids() {
  local docker_host
  docker_host="$(hero_docker_host)"
  # runc and controller mounts are daemon-global.  Even an unrelated or
  # legacy unlabeled container makes a hot switch unsafe, so audit every
  # running container on this dedicated daemon (the function name is retained
  # for compatibility with older callers/tests).
  DOCKER_HOST="${docker_host}" docker ps -q 2>/dev/null
}

hero_safe_cgroup_component() {
  local value="$1"
  value="${value//[^A-Za-z0-9_.-]/_}"
  value="${value:0:96}"
  case "${value}" in ""|.|..) value="run-${value//./_}" ;; esac
  printf '%s' "${value}"
}

hero_select_docker_cgroup_parent() {
  local parent_root="${HERO_LOCAL_DOCKER_CGROUP_PARENT_ROOT:-/hero-docker}"
  local run_component parent
  [[ "${parent_root}" == /* && "${parent_root}" != "/" \
      && "${parent_root}" != *"/../"* && "${parent_root}" != */.. \
      && "${parent_root}" != *"/./"* && "${parent_root}" != */. ]] || return 1
  parent_root="${parent_root%/}"
  run_component="$(hero_safe_cgroup_component "${HERO_RUN_ID:-default}")"
  parent="${HERO_LOCAL_DOCKER_CGROUP_PARENT:-${parent_root}/${run_component}}"
  case "${parent}" in
    "${parent_root}"/*) ;;
    *) return 1 ;;
  esac
  [[ "${parent}" != *"/../"* && "${parent}" != */.. \
      && "${parent}" != *"/./"* && "${parent}" != */. ]] || return 1
  export HERO_LOCAL_DOCKER_CGROUP_PARENT_ROOT="${parent_root}"
  export HERO_LOCAL_DOCKER_CGROUP_PARENT="${parent}"
}

hero_inherit_cpuset_value() {
  local directory="$1" mountpoint="$2" filename="$3"
  local ancestor value=""
  ancestor="${directory%/*}"
  while [[ "${ancestor}" == "${mountpoint}" || "${ancestor}" == "${mountpoint}"/* ]]; do
    if [[ -r "${ancestor}/${filename}" ]]; then
      value="$(<"${ancestor}/${filename}")"
      [[ -n "${value}" ]] && break
    fi
    [[ "${ancestor}" != "${mountpoint}" ]] || break
    ancestor="${ancestor%/*}"
  done
  [[ -n "${value}" ]] || return 1
  printf '%s\n' "${value}" >"${directory}/${filename}"
}

hero_prepare_one_cgroup_parent() {
  local mountpoint="$1" super_options="$2" parent="$3"
  local current="${mountpoint}" component target relative
  local -a components=()
  relative="${parent#/}"
  IFS='/' read -r -a components <<<"${relative}"
  for component in "${components[@]}"; do
    [[ "${component}" =~ ^[A-Za-z0-9_.-]+$ && "${component}" != "." \
        && "${component}" != ".." ]] || return 1
    target="${current}/${component}"
    if [[ ! -d "${target}" ]]; then
      mkdir "${target}" 2>/dev/null || return 1
      _HERO_CGROUP_CREATED_DIRS+=("${target}")
    fi
    if hero_cgroup_super_has_controller "${super_options}" cpuset; then
      [[ -s "${target}/cpuset.cpus" ]] \
        || hero_inherit_cpuset_value "${target}" "${mountpoint}" cpuset.cpus \
        || return 1
      [[ -s "${target}/cpuset.mems" ]] \
        || hero_inherit_cpuset_value "${target}" "${mountpoint}" cpuset.mems \
        || return 1
    fi
    # A limit on the run parent only contains Docker descendants when cgroup
    # v1 hierarchical memory accounting is enabled before containers join it.
    if hero_cgroup_super_has_controller "${super_options}" memory \
        && [[ -e "${target}/memory.use_hierarchy" ]]; then
      [[ "$(<"${target}/memory.use_hierarchy")" == "1" ]] \
        || printf '1\n' >"${target}/memory.use_hierarchy" \
        || return 1
      [[ "$(<"${target}/memory.use_hierarchy")" == "1" ]] || return 1
    fi
    current="${target}"
  done
}

hero_remove_created_cgroup_parents() {
  local index
  for ((index=${#_HERO_CGROUP_CREATED_DIRS[@]} - 1; index >= 0; index--)); do
    rmdir "${_HERO_CGROUP_CREATED_DIRS[${index}]}" >/dev/null 2>&1 || true
  done
  _HERO_CGROUP_CREATED_DIRS=()
}

hero_export_cgroup_parent_files() {
  local parent="$1" index mountpoint super_options directory
  unset HERO_LOCAL_DOCKER_PARENT_MEMORY_LIMIT_FILE
  unset HERO_LOCAL_DOCKER_PARENT_MEMORY_USAGE_FILE
  unset HERO_LOCAL_DOCKER_PARENT_MEMORY_STAT_FILE
  unset HERO_LOCAL_DOCKER_PARENT_PIDS_LIMIT_FILE
  unset HERO_LOCAL_DOCKER_PARENT_PIDS_CURRENT_FILE
  for ((index=0; index<${#_HERO_CGROUP_MOUNTS[@]}; index++)); do
    mountpoint="${_HERO_CGROUP_MOUNTS[${index}]}"
    super_options="${_HERO_CGROUP_SUPER_OPTIONS[${index}]}"
    directory="${mountpoint}${parent}"
    if hero_cgroup_super_has_controller "${super_options}" memory; then
      export HERO_LOCAL_DOCKER_PARENT_MEMORY_LIMIT_FILE="${directory}/memory.limit_in_bytes"
      export HERO_LOCAL_DOCKER_PARENT_MEMORY_USAGE_FILE="${directory}/memory.usage_in_bytes"
      export HERO_LOCAL_DOCKER_PARENT_MEMORY_STAT_FILE="${directory}/memory.stat"
    fi
    if hero_cgroup_super_has_controller "${super_options}" pids; then
      export HERO_LOCAL_DOCKER_PARENT_PIDS_LIMIT_FILE="${directory}/pids.max"
      export HERO_LOCAL_DOCKER_PARENT_PIDS_CURRENT_FILE="${directory}/pids.current"
    fi
  done
  [[ -n "${HERO_LOCAL_DOCKER_PARENT_MEMORY_LIMIT_FILE:-}" \
      && -n "${HERO_LOCAL_DOCKER_PARENT_PIDS_LIMIT_FILE:-}" ]]
}

hero_apply_explicit_cgroup_parent_limits() {
  local requested usage
  requested="${HERO_LOCAL_DOCKER_PARENT_MEMORY_LIMIT_BYTES:-}"
  if [[ -n "${requested}" ]]; then
    [[ "${requested}" =~ ^[1-9][0-9]*$ ]] || return 1
    usage="$(<"${HERO_LOCAL_DOCKER_PARENT_MEMORY_USAGE_FILE}")"
    [[ "${usage}" =~ ^[0-9]+$ ]] || return 1
    if (( requested < usage )); then
      echo "[hero] raising requested Docker parent memory limit to current usage (${usage} bytes)" >&2
      requested="${usage}"
    fi
    printf '%s\n' "${requested}" >"${HERO_LOCAL_DOCKER_PARENT_MEMORY_LIMIT_FILE}" \
      || return 1
    export HERO_LOCAL_DOCKER_PARENT_MEMORY_LIMIT_BYTES="${requested}"
  fi
  requested="${HERO_LOCAL_DOCKER_PARENT_PIDS_LIMIT:-}"
  if [[ -n "${requested}" ]]; then
    [[ "${requested}" == "max" || "${requested}" =~ ^[1-9][0-9]*$ ]] || return 1
    printf '%s\n' "${requested}" >"${HERO_LOCAL_DOCKER_PARENT_PIDS_LIMIT_FILE}" \
      || return 1
  fi
}

hero_set_cgroup_fallback_mode() {
  export HERO_LOCAL_DOCKER_CGROUP_MODE=rootless-fallback
  export HERO_LOCAL_DOCKER_CGROUP_PARENT=""
  unset HERO_LOCAL_DOCKER_PARENT_MEMORY_LIMIT_FILE
  unset HERO_LOCAL_DOCKER_PARENT_MEMORY_USAGE_FILE
  unset HERO_LOCAL_DOCKER_PARENT_MEMORY_STAT_FILE
  unset HERO_LOCAL_DOCKER_PARENT_PIDS_LIMIT_FILE
  unset HERO_LOCAL_DOCKER_PARENT_PIDS_CURRENT_FILE
}

hero_keep_or_install_cgroup_fallback() {
  hero_set_cgroup_fallback_mode
  hero_runc_rootless_wrapper_active && return 0
  if ! hero_install_runc_rootless_wrapper; then
    # Optimisation failures must not turn a usable Docker preflight into a hard
    # launch failure.  Leave the original runc untouched and let Docker report
    # a concrete error later if this host truly requires the wrapper.
    echo "[hero] unable to install cgroup compatibility wrapper; leaving runc unchanged" >&2
  fi
  return 0
}

hero_select_safe_cgroup_fallback() {
  local active_present="${1:-0}"
  if (( active_present )); then
    # A global runc switch is unsafe while stock-runc containers are alive.
    # Keep Docker usable and let the runtime's no-fork /proc monitor enforce
    # the resource fallback for this run.
    hero_set_cgroup_fallback_mode
    echo "[hero] live kernel-cgroup containers detected; leaving stock runc unchanged" >&2
    return 0
  fi
  hero_keep_or_install_cgroup_fallback
}

_hero_prepare_docker_cgroups_locked() {
  local active_ids index mountpoint parent active_present=0
  local driver="${HERO_LOCAL_DOCKER_CGROUP_DRIVER:-cgroupfs}"
  local requested_parent="${HERO_LOCAL_DOCKER_CGROUP_PARENT:-}"
  hero_set_cgroup_fallback_mode
  [[ "${HERO_LOCAL_DOCKER_SKIP_CGROUP_CHECK:-0}" != "1" ]] || return 0
  if [[ "${_HERO_DOCKER_PREFLIGHT_LOCK_HELD:-0}" != "1" ]]; then
    echo "[hero] Docker preflight lock is not held; skipping cgroup/runc mutation" >&2
    return 0
  fi
  if [[ "${driver}" != "cgroupfs" ]]; then
    echo "[hero] Docker cgroup driver ${driver} is not handled by the v1 parent guard; keeping compatibility mode" >&2
    return 0
  fi

  # Never remount controllers or switch runc beneath a live SWE container. If
  # stock runc is already active and every controller is already writable, it
  # is safe to create a sibling per-run parent without mutating the live run.
  if ! active_ids="$(hero_active_swe_container_ids)"; then
    echo "[hero] could not audit live HERO containers; refusing a cgroup/runc hot switch" >&2
    return 0
  fi
  if [[ -n "${active_ids}" ]]; then
    active_present=1
    if hero_runc_rootless_wrapper_active; then
      echo "[hero] live HERO containers use the compatibility wrapper; refusing a cgroup/runc hot switch" >&2
      return 0
    fi
  fi

  if ! hero_collect_cgroup_v1_mounts \
      || ! hero_validate_required_cgroup_controllers; then
    echo "[hero] complete private cgroup v1 hierarchy is unavailable; keeping compatibility mode" >&2
    # Do not install a v1 compatibility wrapper on an unrelated cgroup v2 host.
    hero_runc_rootless_wrapper_active || return 0
    return 0
  fi
  _HERO_CGROUP_REMOUNTED=()
  if (( active_present )); then
    for mountpoint in "${_HERO_CGROUP_MOUNTS[@]}"; do
      if ! hero_cgroup_probe_mount "${mountpoint}"; then
        echo "[hero] live HERO containers detected and cgroup mount is not writable; keeping compatibility mode: ${mountpoint}" >&2
        return 0
      fi
    done
  else
    for ((index=0; index<${#_HERO_CGROUP_MOUNTS[@]}; index++)); do
      if [[ "${_HERO_CGROUP_PRIVATE[${index}]}" != "1" ]]; then
        echo "[hero] cgroup mount is shared; refusing unsafe remount: ${_HERO_CGROUP_MOUNTS[${index}]}" >&2
        return 0
      fi
    done
    for mountpoint in "${_HERO_CGROUP_MOUNTS[@]}"; do
      if ! hero_make_cgroup_mount_writable "${mountpoint}"; then
        echo "[hero] could not make private cgroup mount writable: ${mountpoint}; using compatibility wrapper" >&2
        hero_rollback_cgroup_remounts
        hero_keep_or_install_cgroup_fallback
        return 0
      fi
    done
  fi
  # Probe again only after every mount has been remounted.  This prevents a
  # partial hierarchy from being mistaken for usable kernel enforcement.
  for mountpoint in "${_HERO_CGROUP_MOUNTS[@]}"; do
    if ! hero_cgroup_probe_mount "${mountpoint}"; then
      echo "[hero] cgroup write probe failed after remount: ${mountpoint}; using compatibility wrapper" >&2
      hero_rollback_cgroup_remounts
      hero_select_safe_cgroup_fallback "${active_present}"
      return 0
    fi
  done

  if [[ -n "${requested_parent}" ]]; then
    export HERO_LOCAL_DOCKER_CGROUP_PARENT="${requested_parent}"
  fi
  if ! hero_select_docker_cgroup_parent; then
    echo "[hero] invalid Docker cgroup parent path; using compatibility wrapper" >&2
    hero_rollback_cgroup_remounts
    hero_select_safe_cgroup_fallback "${active_present}"
    return 0
  fi
  parent="${HERO_LOCAL_DOCKER_CGROUP_PARENT}"
  _HERO_CGROUP_CREATED_DIRS=()
  for ((index=0; index<${#_HERO_CGROUP_MOUNTS[@]}; index++)); do
    if ! hero_prepare_one_cgroup_parent \
        "${_HERO_CGROUP_MOUNTS[${index}]}" \
        "${_HERO_CGROUP_SUPER_OPTIONS[${index}]}" "${parent}"; then
      echo "[hero] failed to prepare Docker parent cgroup; using compatibility wrapper" >&2
      hero_remove_created_cgroup_parents
      hero_rollback_cgroup_remounts
      hero_select_safe_cgroup_fallback "${active_present}"
      return 0
    fi
  done
  if ! hero_export_cgroup_parent_files "${parent}" \
      || ! hero_apply_explicit_cgroup_parent_limits; then
    echo "[hero] failed to initialise Docker parent resource files; using compatibility wrapper" >&2
    hero_remove_created_cgroup_parents
    hero_rollback_cgroup_remounts
    hero_select_safe_cgroup_fallback "${active_present}"
    return 0
  fi

  if ! hero_revert_runc_rootless_wrapper; then
    echo "[hero] stock runc verification failed; restoring compatibility wrapper" >&2
    hero_remove_created_cgroup_parents
    hero_rollback_cgroup_remounts
    hero_select_safe_cgroup_fallback "${active_present}"
    return 0
  fi
  export HERO_LOCAL_DOCKER_CGROUP_MODE=kernel
  export HERO_LOCAL_DOCKER_CGROUP_ROOT="${HERO_CGROUP_ROOT:-/sys/fs/cgroup}"
  echo "[hero] kernel cgroup enforcement enabled parent=${parent}" >&2
}

hero_with_docker_preflight_lock() {
  local lock_file="${HERO_LOCAL_DOCKERD_START_LOCK:-/tmp/hero_dockerd_start.lock}"
  local lock_fd return_code=0
  if ! command -v flock >/dev/null 2>&1; then
    echo "[hero] flock is unavailable; skipping cgroup/runc mutation" >&2
    _HERO_DOCKER_PREFLIGHT_LOCK_HELD=0 "$@"
    return
  fi
  exec {lock_fd}>"${lock_file}" || return 1
  if ! flock -w 60 "${lock_fd}"; then
    echo "[hero] timed out waiting for Docker preflight lock" >&2
    exec {lock_fd}>&-
    return 1
  fi
  _HERO_DOCKER_PREFLIGHT_LOCK_HELD=1 \
    _HERO_DOCKER_PREFLIGHT_LOCK_FD="${lock_fd}" \
    "$@" || return_code=$?
  flock -u "${lock_fd}" >/dev/null 2>&1 || true
  exec {lock_fd}>&-
  return "${return_code}"
}

hero_prepare_docker_cgroups() {
  if [[ "${_HERO_DOCKER_PREFLIGHT_LOCK_HELD:-0}" == "1" ]]; then
    _hero_prepare_docker_cgroups_locked
  else
    hero_with_docker_preflight_lock _hero_prepare_docker_cgroups_locked
  fi
}

hero_close_preflight_lock_in_child() {
  # The launcher keeps this dynamic descriptor open until preflight completes.
  # A background dockerd must not inherit it and keep the flock alive forever.
  if [[ "${_HERO_DOCKER_PREFLIGHT_LOCK_FD:-}" =~ ^[0-9]+$ ]]; then
    exec {_HERO_DOCKER_PREFLIGHT_LOCK_FD}>&-
  fi
}

hero_start_overlay2_dockerd() {
  local docker_host data_root log_file storage_driver
  docker_host="$(hero_docker_host)"
  data_root="${HERO_LOCAL_DOCKER_DATA_ROOT:-${XDG_CACHE_HOME:-${HOME}/.cache}/hero/docker-data}"
  storage_driver="${HERO_LOCAL_DOCKER_STORAGE_DRIVER:-overlay2}"
  log_file="${HERO_LOCAL_DOCKERD_LOG:-/tmp/hero-dockerd-${storage_driver}.log}"
  hero_validate_docker_data_root "${data_root}"
  mkdir -p "${data_root}"
  echo "[hero] starting dedicated ${storage_driver} dockerd host=${docker_host} root=${data_root}" >&2
  (
    hero_close_preflight_lock_in_child
    exec nohup dockerd \
      --config-file=/dev/null \
      --data-root="${data_root}" \
      --host="${docker_host}" \
      --storage-driver="${storage_driver}" \
      --iptables=false \
      --bridge=none \
      --userland-proxy=false
  ) >"${log_file}" 2>&1 9>&- &
  for _ in $(seq 1 45); do
    if DOCKER_HOST="${docker_host}" docker info --format '{{.Driver}}' 2>/dev/null \
        | grep -qx "${storage_driver}"; then
      return 0
    fi
    sleep 1
  done
  tail -80 "${log_file}" >&2 2>/dev/null || true
  return 1
}

_hero_check_docker_storage_locked() {
  local docker_host="$1" expected_driver="$2" expected_root="$3"
  local info driver root cgroup_driver cgroup_version
  if ! info="$(docker info --format '{{json .}}' 2>&1)"; then
    if [[ "${HERO_LOCAL_DOCKER_AUTO_START:-1}" != "1" ]]; then
      echo "[hero] Docker daemon unavailable at ${docker_host}: ${info}" >&2
      return 1
    fi
    hero_start_overlay2_dockerd || return 1
    info="$(docker info --format '{{json .}}')" || return 1
  fi
  driver="$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("Driver", ""))' <<<"${info}")"
  root="$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("DockerRootDir", ""))' <<<"${info}")"
  cgroup_driver="$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("CgroupDriver", ""))' <<<"${info}")"
  cgroup_version="$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("CgroupVersion", ""))' <<<"${info}")"
  echo "[hero] Docker storage driver=${driver:-unknown} root=${root:-unknown} cgroup=${cgroup_driver:-unknown}/v${cgroup_version:-unknown}"
  [[ "${driver}" == "${expected_driver}" ]] || {
    echo "[hero] expected driver ${expected_driver}, got ${driver:-unknown}" >&2
    return 1
  }
  [[ "${root}" == "${expected_root}" ]] || {
    echo "[hero] expected root ${expected_root}, got ${root:-unknown}" >&2
    return 1
  }
  export HERO_LOCAL_DOCKER_CGROUP_DRIVER="${cgroup_driver:-unknown}"
  export HERO_LOCAL_DOCKER_CGROUP_VERSION="${cgroup_version:-unknown}"
  _hero_prepare_docker_cgroups_locked
}

hero_check_docker_storage() {
  local docker_host expected_driver expected_root
  docker_host="$(hero_docker_host)"
  expected_driver="${HERO_LOCAL_DOCKER_STORAGE_DRIVER:-overlay2}"
  expected_root="${HERO_LOCAL_DOCKER_DATA_ROOT:-${XDG_CACHE_HOME:-${HOME}/.cache}/hero/docker-data}"
  hero_validate_docker_data_root "${expected_root}"
  export HERO_LOCAL_DOCKER_HOST="${docker_host}"
  export DOCKER_HOST="${docker_host}"
  hero_with_docker_preflight_lock \
    _hero_check_docker_storage_locked "${docker_host}" "${expected_driver}" "${expected_root}"
}

hero_swe_image_ids() {
  local docker_host="$1"
  DOCKER_HOST="${docker_host}" docker images --format '{{.Repository}} {{.ID}}' 2>/dev/null \
    | awk '$1 ~ /(^|\/)(xingyaoww|swebench)\/sweb\.eval/ {print $2}' \
    | sort -u
}

hero_prune_unused_swe_images() {
  local docker_host="$1" image_ids
  image_ids="$(hero_swe_image_ids "${docker_host}")"
  if [[ -n "${image_ids}" ]]; then
    echo "[hero] pruning $(wc -w <<<"${image_ids}") unused SWE image(s)" >&2
    # Docker refuses to remove an image referenced by a running container, so
    # this is safe even when independent runs share the daemon.
    xargs -r env DOCKER_HOST="${docker_host}" docker rmi <<<"${image_ids}" >/dev/null || true
  fi
}

hero_default_prune_unused_swe_images() {
  # VFS workers use the HERO/Slime per-image lease protocol. The legacy bulk
  # prune has no image lease and can race a load or the gap before container
  # creation, so fail closed there. Preserve the established overlay2 default
  # exactly; explicit per-launcher overrides continue to win for both drivers.
  if [[ "${HERO_LOCAL_DOCKER_STORAGE_DRIVER:-overlay2}" == "vfs" ]]; then
    printf '0\n'
  else
    printf '1\n'
  fi
}

hero_cleanup_stale_containers() {
  local docker_host run_id lock_file ids prune_default
  docker_host="$(hero_docker_host)"
  run_id="${HERO_RUN_ID:-}"
  lock_file="${HERO_LOCAL_DOCKER_CLEANUP_LOCK:-/tmp/hero_local_docker_cleanup.lock}"
  prune_default="$(hero_default_prune_unused_swe_images)"
  (
    command -v flock >/dev/null 2>&1 && ! flock -w 60 9 && exit 0
    # Do not remove active containers from other runs on a shared daemon.
    # Recover stopped containers from any previous run, plus every container
    # explicitly owned by this run (useful when reusing HERO_RUN_ID).
    ids="$({
      DOCKER_HOST="${docker_host}" docker ps -aq \
        --filter 'label=ai.hero.namespace=swegym-local-docker' \
        --filter 'status=exited' 2>/dev/null
      DOCKER_HOST="${docker_host}" docker ps -aq \
        --filter 'label=ai.hero.namespace=swegym-local-docker' \
        --filter 'status=dead' 2>/dev/null
      if [[ -n "${run_id}" ]]; then
        DOCKER_HOST="${docker_host}" docker ps -aq \
          --filter 'label=ai.hero.namespace=swegym-local-docker' \
          --filter "label=ai.hero.run_id=${run_id}" 2>/dev/null
      fi
    } | sort -u)"
    if [[ -n "${ids}" ]]; then
      echo "[hero] removing stale HERO SWE containers: $(wc -w <<<"${ids}")"
      xargs -r env DOCKER_HOST="${docker_host}" docker rm -f <<<"${ids}" >/dev/null
    fi
    if [[ "${HERO_LOCAL_DOCKER_PREFLIGHT_PRUNE_UNUSED_SWE_IMAGES:-${prune_default}}" == "1" ]]; then
      hero_prune_unused_swe_images "${docker_host}"
    fi
  ) 9>"${lock_file}"
}

hero_cleanup_run_containers() {
  local docker_host run_id lock_file ids prune_default
  docker_host="$(hero_docker_host)"
  run_id="${HERO_RUN_ID:-}"
  lock_file="${HERO_LOCAL_DOCKER_CLEANUP_LOCK:-/tmp/hero_local_docker_cleanup.lock}"
  prune_default="$(hero_default_prune_unused_swe_images)"
  [[ -n "${run_id}" ]] || return 0
  (
    if command -v flock >/dev/null 2>&1 && ! flock -w 60 9; then
      echo "[hero] Docker exit cleanup timed out waiting for ${lock_file}" >&2
      exit 0
    fi
    ids="$(DOCKER_HOST="${docker_host}" docker ps -aq \
      --filter 'label=ai.hero.namespace=swegym-local-docker' \
      --filter "label=ai.hero.run_id=${run_id}" 2>/dev/null)"
    if [[ -n "${ids}" ]]; then
      echo "[hero] removing containers for run ${run_id}" >&2
      xargs -r env DOCKER_HOST="${docker_host}" docker rm -f <<<"${ids}" >/dev/null || true
    fi
    if [[ "${HERO_LOCAL_DOCKER_EXIT_PRUNE_UNUSED_SWE_IMAGES:-${prune_default}}" == "1" ]]; then
      hero_prune_unused_swe_images "${docker_host}"
    fi
  ) 9>"${lock_file}"
}
