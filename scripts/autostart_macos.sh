#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This script only supports macOS (launchd)." >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="${NSB_LAUNCH_LABEL:-com.noslouchbench.agent}"
PLIST_PATH="$HOME/Library/LaunchAgents/${LABEL}.plist"
LOG_DIR="$ROOT_DIR/outputs/startup"
UID_VALUE="$(id -u)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
RUNNER_SCRIPT="$LOG_DIR/run_webcam_runner.sh"

if [[ ! -x "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "python3 not found. Install Python or create .venv first." >&2
    exit 1
  fi
fi

xml_escape() {
  local raw="${1//&/&amp;}"
  raw="${raw//</&lt;}"
  raw="${raw//>/&gt;}"
  printf '%s' "$raw"
}

default_args=(
  run-webcam
  --model yolo-pose
  --camera-id 0
  --screen-blocker
)

shell_join() {
  local out=""
  local arg=""
  for arg in "$@"; do
    out+=$(printf '%q ' "$arg")
  done
  printf '%s' "${out% }"
}

build_runner_script() {
  local -a cmd_args=("$@")
  local -a full_cmd=("$PYTHON_BIN" -m noslouchbench.cli "${cmd_args[@]}")
  mkdir -p "$LOG_DIR"

  cat > "$RUNNER_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd $(printf '%q' "$ROOT_DIR")
export PYTHONPATH=src
while true; do
  $(shell_join "${full_cmd[@]}") >> $(printf '%q' "$LOG_DIR/stdout.log") 2>> $(printf '%q' "$LOG_DIR/stderr.log")
  sleep 2
done
EOF
  chmod +x "$RUNNER_SCRIPT"
}

build_plist() {
  local runner_cmd="/bin/bash $(printf '%q' "$RUNNER_SCRIPT")"
  mkdir -p "$HOME/Library/LaunchAgents"
  mkdir -p "$LOG_DIR"

  {
    cat <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$(xml_escape "$LABEL")</string>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <false/>
  <key>WorkingDirectory</key>
  <string>$(xml_escape "$ROOT_DIR")</string>
  <key>ProgramArguments</key>
  <array>
    <string>/usr/bin/osascript</string>
    <string>-e</string>
    <string>tell application "Terminal" to do script "$(xml_escape "$runner_cmd")"</string>
    <string>-e</string>
    <string>tell application "Terminal" to hide</string>
  </array>
  <key>StandardOutPath</key>
  <string>$(xml_escape "$LOG_DIR/stdout.log")</string>
  <key>StandardErrorPath</key>
  <string>$(xml_escape "$LOG_DIR/stderr.log")</string>
</dict>
</plist>
EOF
  } > "$PLIST_PATH"
}

stop_job() {
  launchctl bootout "gui/${UID_VALUE}/${LABEL}" >/dev/null 2>&1 || true
  pkill -f "$RUNNER_SCRIPT" >/dev/null 2>&1 || true
}

start_job() {
  if [[ ! -f "$PLIST_PATH" ]]; then
    echo "LaunchAgent not installed: $PLIST_PATH" >&2
    exit 1
  fi
  stop_job
  launchctl bootstrap "gui/${UID_VALUE}" "$PLIST_PATH"
  launchctl enable "gui/${UID_VALUE}/${LABEL}" >/dev/null 2>&1 || true
}

status_job() {
  if launchctl print "gui/${UID_VALUE}/${LABEL}" >/dev/null 2>&1; then
    echo "NoSlouchBench autostart is loaded (${LABEL})."
    launchctl print "gui/${UID_VALUE}/${LABEL}" | sed -n '1,25p'
    if pgrep -fl "$RUNNER_SCRIPT" >/dev/null 2>&1; then
      echo ""
      echo "Runner process:"
      pgrep -fl "$RUNNER_SCRIPT" | sed -n '1,5p'
    else
      echo "Runner process not found."
    fi
  else
    echo "NoSlouchBench autostart is not loaded (${LABEL})."
    if [[ -f "$PLIST_PATH" ]]; then
      echo "Plist exists at: $PLIST_PATH"
    fi
  fi
}

usage() {
  cat <<EOF
Usage:
  scripts/autostart_macos.sh install [-- <run-webcam args>]
  scripts/autostart_macos.sh start
  scripts/autostart_macos.sh stop
  scripts/autostart_macos.sh status
  scripts/autostart_macos.sh uninstall

Examples:
  scripts/autostart_macos.sh install
  scripts/autostart_macos.sh install -- --camera-id 1 --model yolo-pose --screen-blocker
  scripts/autostart_macos.sh stop

Notes:
  - install creates ~/Library/LaunchAgents/${LABEL}.plist
  - default run args: run-webcam --model yolo-pose --camera-id 0 --screen-blocker
  - Camera access uses a Terminal-hosted runner (hidden after launch).
  - The runner auto-restarts the webcam command every 2 seconds if it exits.
EOF
}

cmd="${1:-}"
case "$cmd" in
  install)
    shift || true
    run_args=("${default_args[@]}")
    if [[ "${1:-}" == "--" ]]; then
      shift
      if [[ $# -gt 0 ]]; then
        run_args=("$@")
        if [[ "${run_args[0]}" == -* ]]; then
          run_args=(run-webcam "${run_args[@]}")
        fi
      fi
    fi
    build_runner_script "${run_args[@]}"
    build_plist
    start_job
    echo "Installed and started LaunchAgent: $PLIST_PATH"
    ;;
  start)
    start_job
    echo "Started LaunchAgent: ${LABEL}"
    ;;
  stop)
    stop_job
    echo "Stopped LaunchAgent: ${LABEL}"
    ;;
  status)
    status_job
    ;;
  uninstall)
    stop_job
    rm -f "$PLIST_PATH"
    echo "Uninstalled LaunchAgent: $PLIST_PATH"
    ;;
  *)
    usage
    exit 1
    ;;
esac
