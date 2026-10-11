#!/bin/bash
# Wait for a child once; retry only when a signal interrupted wait(2).
wait_child() {
  local pid="$1" wait_rc
  while true; do
    wait_rc=0
    wait "$pid" || wait_rc=$?
    if kill -0 "$pid" 2>/dev/null; then
      continue
    fi
    return "$wait_rc"
  done
}
