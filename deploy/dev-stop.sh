#!/bin/bash

echo "Stop running docker compose"
cd A2rchi-dev/A2rchi/deploy/
docker compose -f dev-compose.yaml down
