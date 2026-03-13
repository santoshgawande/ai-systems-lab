# Deploy

Homelab deployment configs for the AI systems lab.

## Servers

| Server | IP | Stack | README |
|---|---|---|---|
| proxmox1 | 192.168.0.111 | PostgreSQL + pgvector, Redis | [proxmox1/README.md](proxmox1/README.md) |
| proxmox2 | 192.168.0.112 | Jupyter Lab, Qdrant | [proxmox2/README.md](proxmox2/README.md) |

## Mac Studio
- Ollama running at `http://localhost:11434`
- Models: `llama3`, `deepseek-r1`, `qwen3-coder:30b`, `glm-4.7-flash`, `nomic-embed-text`

## Quick redeploy (from Mac Studio)
```bash
# proxmox1
scp deploy/proxmox1/docker-compose.yml root@proxmox1:~/docker-compose.yml
ssh root@proxmox1 "docker compose up -d"

# proxmox2
scp deploy/proxmox2/docker-compose.yml root@proxmox2:~/docker-compose.yml
ssh root@proxmox2 "docker compose up -d"
```

## Common issues
- **apt fails on proxmox1**: enterprise repo blocked — re-run step 2 in proxmox1/README.md
- **apt fails on proxmox2**: DNS broken — run `echo 'nameserver 8.8.8.8' > /etc/resolv.conf`
- **postgres restarts**: needs `privileged: true` for Unix socket — already in compose file
- **jupyter restarts**: same, already in compose file
