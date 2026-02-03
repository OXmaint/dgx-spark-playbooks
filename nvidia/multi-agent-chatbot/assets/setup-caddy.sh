#!/bin/bash
# Setup script for Caddy reverse proxy with TLS
# Installs Caddy on the host and configures it to proxy to Docker services

set -e

echo "=========================================="
echo "Setting up Caddy reverse proxy with TLS"
echo "=========================================="

# Install Caddy
echo ""
echo "Installing Caddy..."
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https curl
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update
sudo apt install -y caddy

echo ""
echo "Caddy installed successfully!"

# Create Caddyfile
echo ""
echo "Creating Caddyfile..."

sudo tee /etc/caddy/Caddyfile > /dev/null <<'EOF'
# Caddyfile - Reverse proxy with TLS for local services
# Uses self-signed internal certificates for localhost

{
    # Enable local/internal TLS certificates
    local_certs
}

# n8n - https://localhost (port 443)
localhost {
    tls internal

    reverse_proxy localhost:5678

    header {
        Strict-Transport-Security "max-age=31536000; includeSubDomains"
        X-Content-Type-Options "nosniff"
        X-Frame-Options "SAMEORIGIN"
        X-XSS-Protection "1; mode=block"
        Permissions-Policy "interest-cohort=()"
    }
}

# Backend API - https://localhost:8443
localhost:8443 {
    tls internal

    reverse_proxy localhost:8000
}

# Frontend - https://localhost:3443
localhost:3443 {
    tls internal

    reverse_proxy localhost:3000
}

# DeepSeek OCR - https://localhost:8543
localhost:8543 {
    tls internal

    reverse_proxy localhost:8100
}
EOF

echo "Caddyfile created at /etc/caddy/Caddyfile"

# Restart Caddy to apply configuration
echo ""
echo "Restarting Caddy service..."
sudo systemctl restart caddy
sudo systemctl enable caddy

# Check status
echo ""
echo "Caddy status:"
sudo systemctl status caddy --no-pager

echo ""
echo "=========================================="
echo "Caddy setup complete!"
echo "=========================================="
echo ""
echo "HTTPS endpoints available:"
echo "  - n8n:          https://localhost"
echo "  - Backend API:  https://localhost:8443"
echo "  - Frontend:     https://localhost:3443"
echo "  - DeepSeek OCR: https://localhost:8543"
echo ""
echo "Note: Your browser will show a certificate warning for"
echo "self-signed certificates. Click 'Advanced' -> 'Proceed'"
echo "to continue. This is expected for local development."
echo ""
