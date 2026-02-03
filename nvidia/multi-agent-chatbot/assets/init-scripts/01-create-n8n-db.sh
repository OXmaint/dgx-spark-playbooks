#!/bin/bash
set -e

# Create n8n database and user
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE USER $N8N_USER WITH PASSWORD '$N8N_PASSWORD';
    CREATE DATABASE $N8N_DB OWNER $N8N_USER;
    GRANT ALL PRIVILEGES ON DATABASE $N8N_DB TO $N8N_USER;
EOSQL

echo "n8n database and user created successfully"
