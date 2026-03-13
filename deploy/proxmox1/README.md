# proxmox1 (192.168.0.111) — Services Stack

Runs: PostgreSQL + pgvector, Redis

## First-Time Setup

### 1. SSH into proxmox1
```bash
ssh root@proxmox1
```

### 2. Fix apt sources (disable enterprise, enable free)
```bash
echo '# disabled' > /etc/apt/sources.list.d/pve-enterprise.sources
echo '# disabled' > /etc/apt/sources.list.d/ceph.sources

cat > /etc/apt/sources.list.d/pve-no-sub.sources << 'EOF'
Types: deb
URIs: http://download.proxmox.com/debian/pve
Suites: trixie
Components: pve-no-subscription
Signed-By: /usr/share/keyrings/proxmox-archive-keyring.gpg
EOF
```

### 3. Install Docker
```bash
apt-get update && apt-get install -y docker.io docker-compose
systemctl enable docker && systemctl start docker
```

### 4. Deploy the stack
```bash
# From Mac Studio — copy the compose file
scp deploy/proxmox1/docker-compose.yml root@proxmox1:~/docker-compose.yml

# On proxmox1
docker compose up -d
```

### 5. Verify
```bash
docker ps
# Should show postgres and redis both Up

docker exec root-postgres-1 psql -U ai -d ailab -c 'CREATE EXTENSION IF NOT EXISTS vector; SELECT version();'
# Should print PostgreSQL version
```

## Redeploy (after changes)
```bash
scp deploy/proxmox1/docker-compose.yml root@proxmox1:~/docker-compose.yml
ssh root@proxmox1 "docker compose up -d"
```

## Wipe and start fresh
```bash
ssh root@proxmox1 "docker compose down -v && docker compose up -d"
```

## Services

| Service | Port | Credentials |
|---|---|---|
| PostgreSQL + pgvector | 5432 | user: `ai` / pass: `ai` / db: `ailab` |
| Redis | 6379 | no auth |

## Connect from Mac Studio
```bash
# Postgres
psql -h proxmox1 -U ai -d ailab

# Redis
redis-cli -h proxmox1
```

## Notes
- `privileged: true` is required on postgres — Proxmox blocks Unix socket creation in containers without it
- DNS on proxmox1 uses `8.8.8.8` — if apt stops working, check `/etc/resolv.conf`
