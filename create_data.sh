#!/bin/bash

# Create folders
mkdir -p history/full history/tradable

# Copy file to folders
cp -f shioaji_future_TXFR1_1T.pkl history/full
cp -f shioaji_future_TXFR1_1T.pkl history/tradable
docker-compose run --rm database python3 settlement.py
echo "Done!"