# Deploying HoopsAI for free

HoopsAI runs for **$0** on one **Oracle Cloud Always Free** virtual machine. The whole Docker Compose stack runs there, with HTTPS from Caddy and a free DuckDNS name. Backups go to Oracle Object Storage.

Free tiers change, so re-check these limits once a year.

| Need | Free service | Limit (Oracle, check at sign-up) |
|---|---|---|
| Server | Ampere A1 Flex VM (ARM64) | 4 OCPU, 24 GB RAM |
| Disk | Boot/block volume | 200 GB in total |
| Backups | Object Storage | 20 GB |
| Domain | DuckDNS subdomain | free |
| Uptime alerts | UptimeRobot | 50 monitors, 5-minute checks |

**Why not the usual free hosts?**
- **Neon and Supabase:** the free tiers allow about 0.5 GB, but the database is 2.8 GB. Supabase also pauses inactive projects.
- **Render:** the free Postgres expires after 30 days, and free web services sleep, which would cut live WebSockets.
- **Live and worker services:** they must run around the clock, which rules out hosts whose free tier sleeps.

## 1. Oracle account and VM
1. Sign up at oracle.com/cloud/free. A card is needed for identity verification only.
   - Pick your **home region** carefully, because it can't be changed. A region with A1 capacity matters most.
   - If creating the VM later fails with "out of capacity", retry at another time of day.
2. **Billing → Upgrade to Pay As You Go.** Always Free resources stay free, and Oracle then stops reclaiming idle Always Free VMs, which is what keeps the site up for years.
3. **Billing → Budgets:** create a **$1 budget with an email alert at 1%.** Only ever create resources marked "Always Free-eligible".
4. **Compute → Create instance:**
   - Image: Ubuntu 24.04 (aarch64)
   - Shape: VM.Standard.A1.Flex, 4 OCPU, 24 GB
   - Boot volume: 100 GB
   - Add your SSH public key.
5. **Networking:** in the subnet's security list, allow inbound TCP **22, 80 and 443** only. On the VM:
   ```sh
   sudo iptables -I INPUT 6 -p tcp -m multiport --dports 80,443 -j ACCEPT && sudo netfilter-persistent save
   ```

## 2. Gate check: can the VM reach stats.nba.com?
Run this before installing anything else. stats.nba.com blocks many cloud IP ranges, and the result decides where data updates run.
```sh
sudo apt update && sudo apt install -y python3-pip && pip install --user --break-system-packages nba_api
python3 -c "from nba_api.stats.endpoints import scoreboardv3; print(len(scoreboardv3.ScoreboardV3(game_date='2026-10-20', league_id='00', timeout=15).get_dict()['scoreboard']['games']), 'games')"
```
- **Prints a game count:** everything runs on the VM. Skip section 7.
- **Times out or returns 403:** use the PC fallback in section 7. The website still runs on the VM around the clock.

## 3. Install and configure
```sh
curl -fsSL https://get.docker.com | sh && sudo usermod -aG docker $USER   # log out and back in
git clone https://github.com/royce123541/HoopsAI.git && cd HoopsAI
cp .env.example .env
```
Edit `.env`:
- `POSTGRES_PASSWORD`: a long random string. Change it in `HOOPSAI_DATABASE_URL` as well.
- `DOMAIN=<name>.duckdns.org`

**Domain:**
1. Create the subdomain at duckdns.org, pointing at the VM's public IP.
2. Add the DuckDNS update cron line it gives you, which keeps the IP current.

## 4. Move the data from the PC
On the PC:
```sh
docker compose exec -T db pg_dump -U hoopsai -Fc hoopsai > hoopsai.dump
docker run --rm -v hoopsai_mlflow:/mlflow:ro -v "$PWD":/out alpine tar czf /out/mlflow.tgz -C /mlflow .
scp hoopsai.dump mlflow.tgz ubuntu@<vm-ip>:~/HoopsAI/
```
On the VM:
```sh
P="-f docker-compose.yml -f deploy/docker-compose.prod.yml"
docker compose $P up -d db mlflow && sleep 10
docker compose $P exec -T db pg_restore -U hoopsai -d hoopsai --clean --if-exists < hoopsai.dump
docker compose $P stop mlflow
docker run --rm -v hoopsai_mlflow:/mlflow -v "$PWD":/in alpine sh -c "rm -rf /mlflow/* && tar xzf /in/mlflow.tgz -C /mlflow"
docker compose $P up -d --build
```
Open `https://<name>.duckdns.org`. Caddy obtains the TLS certificate on the first request.

## 5. Backups
1. **Storage → Buckets:** create `hoopsai-backups`. Add a lifecycle rule that deletes objects after 30 days, which keeps usage well under 20 GB.
2. Install and configure the OCI CLI (`oci setup config`).
3. Add `deploy/backup.sh` to cron:
   ```
   30 9 * * * /home/ubuntu/HoopsAI/deploy/backup.sh >> /home/ubuntu/backup.log 2>&1
   ```
   It dumps the database nightly and the MLflow models on Mondays.

**Restore drill (once a season):**
1. Download a dump with `oci os object get`.
2. Run `createdb` for a scratch database, then `pg_restore` the dump into it.
3. Check the games count.

## 6. Keep it alive for years
- **Uptime:** an UptimeRobot HTTP monitor on `https://<domain>/api/health`, with email alerts.
- **Security updates:** `sudo apt install unattended-upgrades` applies them automatically.
- **Deploys:** after pushing to GitHub, run `./deploy/update.sh` on the VM. It pulls, rebuilds and restarts.
- **Yearly:** re-read the Oracle Always Free terms, check that the budget alert never fired, and run the restore drill.

## 7. Fallback: data jobs on the PC
Use this only if section 2 failed.
1. On the VM: `docker compose $P stop worker live`, and set `restart: "no"` for those two services in a local override so they don't come back.
2. On the PC, keep an SSH tunnel to the VM's loopback-only Postgres, Redis and MLflow:
   ```sh
   autossh -M 0 -N -L 15432:127.0.0.1:5432 -L 16379:127.0.0.1:6379 -L 15000:127.0.0.1:5000 ubuntu@<vm-ip>
   ```
3. On the PC: `docker compose -f deploy/docker-compose.pc.yml --env-file .env up -d --build`.

The site stays up all the time. Scores, predictions and live charts update whenever the PC is on. The next nightly run catches up on any days that were missed.
