#!/usr/bin/env sh
# Deploy the latest main on the VM: pull, rebuild changed images, restart what changed.
set -eu
cd "$(dirname "$0")/.."
git pull --ff-only
docker compose -f docker-compose.yml -f deploy/docker-compose.prod.yml up -d --build
docker image prune -f
docker compose -f docker-compose.yml -f deploy/docker-compose.prod.yml ps
