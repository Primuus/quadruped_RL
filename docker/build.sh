#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
docker compose up -d --build
docker compose exec quadruped_train bash
