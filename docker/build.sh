#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
docker compose up -d --build
if [ "$#" -gt 0 ]; then
  docker compose exec quadruped_train "$@"
else
  docker compose exec quadruped_train bash
fi
