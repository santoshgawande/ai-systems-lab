# proxmox2 (192.168.0.112) — AI Workloads Stack

Runs: Jupyter Lab, Qdrant

## First-Time Setup

### 1. SSH into proxmox2
```bash
ssh root@proxmox2
```

### 2. Fix DNS (proxmox2 defaults to 127.0.0.1 which doesn't resolve)
```bash
echo 'nameserver 8.8.8.8' > /etc/resolv.conf
echo 'nameserver 1.1.1.1' >> /etc/resolv.conf
```

### 3. Fix apt sources (disable enterprise, enable free)
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

### 4. Install Docker
```bash
apt-get update && apt-get install -y docker.io docker-compose
systemctl enable docker && systemctl start docker
```

### 5. Deploy the stack
```bash
# From Mac Studio — copy the compose file
scp deploy/proxmox2/docker-compose.yml root@proxmox2:~/docker-compose.yml

# On proxmox2 — create notebooks directory first
mkdir -p ~/notebooks
docker compose up -d
```

### 6. Verify
```bash
docker ps
# Should show jupyter and qdrant both Up

# Get Jupyter token
docker logs root-jupyter-1 2>&1 | grep token | tail -2
```

## Redeploy (after changes)
```bash
scp deploy/proxmox2/docker-compose.yml root@proxmox2:~/docker-compose.yml
ssh root@proxmox2 "docker compose up -d"
```

## Wipe and start fresh
```bash
ssh root@proxmox2 "docker compose down -v && docker compose up -d"
```

## Services

| Service | Port | Access |
|---|---|---|
| Jupyter Lab | 8888 | `http://proxmox2:8888` — get token from logs |
| Qdrant | 6333 | `http://proxmox2:6333` |

## Get Jupyter token
```bash
ssh root@proxmox2 "docker logs root-jupyter-1 2>&1 | grep token | tail -2"
```

Then open: `http://proxmox2:8888/lab?token=<token>`

## Connect Jupyter to Mac Studio Ollama
In any notebook:
```python
import requests
resp = requests.post("http://192.168.0.1:11434/api/generate",  # Mac Studio IP
    json={"model": "llama3", "prompt": "Hello", "stream": False})
print(resp.json()["response"])
```

## Notes
- `privileged: true` is required on jupyter — Proxmox blocks Unix socket creation in containers without it
- DNS must be set to `8.8.8.8` — proxmox2 defaults to `127.0.0.1` which fails to resolve external hostnames
- Qdrant does not need privileged mode
