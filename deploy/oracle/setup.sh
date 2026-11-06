#!/usr/bin/env bash
# One-time setup (and later updates) of the wake-word web app on an Oracle Cloud Always Free Ubuntu server.
#
#   curl -fsSL https://raw.githubusercontent.com/POOJASREE-B/Wake-word-Recognition-for-People-who-Stutter/main/deploy/oracle/setup.sh | sudo bash -s -- your-name.duckdns.org
#
# What it does:
#   1. installs Python, FFmpeg, Git and the Caddy web server
#   2. downloads the project into /opt/wakeword/app and installs it in /opt/wakeword/venv (no PyTorch: ONNX Runtime)
#   3. runs the app as a systemd service that restarts on failure and starts at boot
#   4. adds a timer that calls /api/health every 10 minutes, so the model is always loaded in memory
#      (Oracle treats a server as idle only if CPU, network AND memory all stay under 20 % for 7 days)
#   5. configures Caddy to serve the app on https://<your domain> with an automatic Let's Encrypt certificate
#   6. opens ports 80 and 443 in the server's own firewall
# Running it again pulls the latest code from GitHub and restarts the app.
set -euo pipefail

DOMAIN="${1:-}"
REPO="${REPO:-https://github.com/POOJASREE-B/Wake-word-Recognition-for-People-who-Stutter.git}"
BASE=/opt/wakeword
APP="$BASE/app"
VENV="$BASE/venv"
SVC_USER=wakeword

if [ -z "$DOMAIN" ]; then echo "usage: sudo bash setup.sh your-name.duckdns.org" >&2; exit 1; fi
if [ "$(id -u)" -ne 0 ]; then echo "run this script with sudo" >&2; exit 1; fi

echo "== 1/6 system packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y python3 python3-venv python3-pip git ffmpeg curl gnupg \
    debian-keyring debian-archive-keyring apt-transport-https iptables-persistent

if ! command -v caddy >/dev/null 2>&1; then
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
        | gpg --dearmor --yes -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
        > /etc/apt/sources.list.d/caddy-stable.list
    apt-get update -y
    apt-get install -y caddy
fi

echo "== 2/6 project code"
id -u "$SVC_USER" >/dev/null 2>&1 || useradd --system --create-home --home-dir "$BASE" --shell /usr/sbin/nologin "$SVC_USER"
mkdir -p "$BASE" && chown "$SVC_USER:$SVC_USER" "$BASE"
if [ -d "$APP/.git" ]; then
    sudo -u "$SVC_USER" git -C "$APP" pull --ff-only
else
    sudo -u "$SVC_USER" git clone --depth 1 "$REPO" "$APP"
fi

echo "== 3/6 Python environment"
[ -x "$VENV/bin/python" ] || sudo -u "$SVC_USER" python3 -m venv "$VENV"
sudo -u "$SVC_USER" "$VENV/bin/python" -m pip install --upgrade pip
sudo -u "$SVC_USER" "$VENV/bin/python" -m pip install -r "$APP/requirements.txt"
(cd "$APP" && sudo -u "$SVC_USER" "$VENV/bin/python" run.py check)

echo "== 4/6 app service and keep-loaded timer"
cat > /etc/systemd/system/wakeword.service <<EOF
[Unit]
Description=Wake-word detection web app
After=network-online.target
Wants=network-online.target

[Service]
User=$SVC_USER
WorkingDirectory=$APP
Environment=PYTHONUNBUFFERED=1
ExecStart=$VENV/bin/python -m waitress --host=127.0.0.1 --port=8000 --threads=4 --call server.app:create_app
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

cat > /etc/systemd/system/wakeword-warm.service <<'EOF'
[Unit]
Description=Load the wake-word model (keeps it in memory)
After=wakeword.service

[Service]
Type=oneshot
ExecStart=/usr/bin/curl -fsS -o /dev/null --max-time 120 http://127.0.0.1:8000/api/health
EOF

cat > /etc/systemd/system/wakeword-warm.timer <<'EOF'
[Unit]
Description=Call the wake-word health check every 10 minutes

[Timer]
OnBootSec=2min
OnUnitActiveSec=10min

[Install]
WantedBy=timers.target
EOF

echo "== 5/6 HTTPS with Caddy"
cat > /etc/caddy/Caddyfile <<EOF
$DOMAIN {
    encode gzip
    reverse_proxy 127.0.0.1:8000
}
EOF

echo "== 6/6 firewall (ports 80 and 443)"
for port in 80 443; do
    if ! iptables -C INPUT -p tcp -m state --state NEW --dport "$port" -j ACCEPT 2>/dev/null; then
        n=$(iptables -L INPUT --line-numbers | awk '/REJECT/ {print $1; exit}')
        if [ -n "$n" ]; then
            iptables -I INPUT "$n" -p tcp -m state --state NEW --dport "$port" -j ACCEPT
        else
            iptables -A INPUT -p tcp -m state --state NEW --dport "$port" -j ACCEPT
        fi
    fi
done
netfilter-persistent save

systemctl daemon-reload
systemctl enable wakeword.service wakeword-warm.timer
systemctl restart wakeword.service
systemctl start wakeword-warm.timer
systemctl restart caddy
sleep 8
systemctl start wakeword-warm.service || true

echo
echo "Done. Open https://$DOMAIN"
echo "Status:  sudo systemctl status wakeword   |   Logs:  sudo journalctl -u wakeword -f"
