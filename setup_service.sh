#!/bin/bash

# --- CONFIGURATION ---
REPO_DIR="$HOME/echo-tts"  # Make sure this matches where you cloned the repo
ENV_NAME="echo-tts"
SERVICE_NAME="echo-tts-api"
PYTHON_SCRIPT="api.py"     # The new file we created
# ---------------------

# Get current username
CURRENT_USER=$(whoami)

# Find Conda
if [ -d "$HOME/miniconda" ]; then
    CONDA_PATH="$HOME/miniconda"
elif [ -d "$HOME/miniconda3" ]; then
    CONDA_PATH="$HOME/miniconda3"
elif [ -d "$HOME/anaconda3" ]; then
    CONDA_PATH="$HOME/anaconda3"
else
    # Fallback try
    CONDA_PATH=$(conda info --base 2>/dev/null)
    if [ -z "$CONDA_PATH" ]; then
        echo "Error: Could not find Conda installation path."
        exit 1
    fi
fi

echo "Creating Systemd Service..."

# Create the service file in /tmp
cat > /tmp/${SERVICE_NAME}.service << EOF
[Unit]
Description=Echo TTS Production API
After=network.target

[Service]
Type=simple
User=$CURRENT_USER
Group=$CURRENT_USER

# Set working directory to the repo
WorkingDirectory=$REPO_DIR

# 1. Source Conda
# 2. Activate Environment
# 3. Run API
ExecStart=/bin/bash -c 'source $CONDA_PATH/etc/profile.d/conda.sh && \
                        conda activate $ENV_NAME && \
                        python $PYTHON_SCRIPT'

# Restart logic: Always restart, wait 5s between attempts
Restart=always
RestartSec=5

# Output logs to syslog (viewable via journalctl)
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Move and Enable
echo "Installing service to /etc/systemd/system/..."
sudo mv /tmp/${SERVICE_NAME}.service /etc/systemd/system/${SERVICE_NAME}.service

sudo chown root:root /etc/systemd/system/${SERVICE_NAME}.service
sudo chmod 644 /etc/systemd/system/${SERVICE_NAME}.service

echo "Reloading Daemon..."
sudo systemctl daemon-reload
sudo systemctl enable $SERVICE_NAME

echo "---------------------------------------"
echo "Service Installed: $SERVICE_NAME"
echo ""
echo "Start it now:      sudo systemctl start $SERVICE_NAME"
echo "Check status:      sudo systemctl status $SERVICE_NAME"
echo "View logs:         sudo journalctl -u $SERVICE_NAME -f"