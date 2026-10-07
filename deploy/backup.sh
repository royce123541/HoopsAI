#!/usr/bin/env sh
# Nightly off-site backup (cron: 30 9 * * * /home/ubuntu/HoopsAI/deploy/backup.sh >> ~/backup.log 2>&1)
# Uploads to an Oracle Object Storage bucket; a bucket lifecycle rule deletes objects after
# 30 days, keeping usage well under the 20 GB free tier. Requires the configured `oci` CLI.
set -eu
cd "$(dirname "$0")/.."
BUCKET="${HOOPSAI_BACKUP_BUCKET:-hoopsai-backups}"
STAMP=$(date -u +%Y%m%d)
DIR=$(mktemp -d)
trap 'rm -rf "$DIR"' EXIT

docker compose exec -T db pg_dump -U "${POSTGRES_USER:-hoopsai}" -Fc "${POSTGRES_DB:-hoopsai}" > "$DIR/hoopsai-$STAMP.dump"
oci os object put --bucket-name "$BUCKET" --file "$DIR/hoopsai-$STAMP.dump" --force

# Models and registry change at most weekly; back them up on Mondays.
if [ "$(date -u +%u)" = "1" ]; then
  docker run --rm -v hoopsai_mlflow:/mlflow:ro -v "$DIR":/out alpine tar czf "/out/mlflow-$STAMP.tgz" -C /mlflow .
  oci os object put --bucket-name "$BUCKET" --file "$DIR/mlflow-$STAMP.tgz" --force
fi
echo "backup $STAMP ok"
