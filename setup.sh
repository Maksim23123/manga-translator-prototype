#!/bin/bash
# Setup script for Mac/Linux

set -e

# Step 1: Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists."
fi

# Step 2: Install requirements
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    ./venv/bin/pip install -r requirements.txt
else
    echo "requirements.txt not found."
fi

# Step 3: Copy config.template.json to config.json if not present
if [ -f "config.template.json" ]; then
    if [ ! -f "config.json" ]; then
        echo "Copying config.template.json to config.json..."
        cp config.template.json config.json
    else
        echo "config.json already exists."
    fi
else
    echo "config.template.json not found."
fi

# Step 4: Ensure inputs and outputs folders exist
for folder in inputs outputs; do
    if [ ! -d "$folder" ]; then
        echo "Creating folder: $folder"
        mkdir -p "$folder"
    else
        echo "Folder already exists: $folder"
    fi
done

echo "Setup complete."
