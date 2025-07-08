#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- 1. Check Environment Variables ---
echo "🔎 Checking for required environment variables..."
missing_vars=()

# Check each required variable
[[ -z "$WANDB_API_KEY" ]] && missing_vars+=("WANDB_API_KEY")
[[ -z "$GITHUB_NAME" ]]   && missing_vars+=("GITHUB_NAME")
[[ -z "$GITHUB_EMAIL" ]]  && missing_vars+=("GITHUB_EMAIL")

if (( ${#missing_vars[@]} > 0 )); then
    echo -e "\nThe following required environment variables are NOT set:"
    for var in "${missing_vars[@]}"; do
        echo "  - $var"
    done
    echo -e "\nPlease export them before proceeding.\n"
    exit 1
else
    echo -e "\n✅ All required environment variables are set. Continuing...\n"
fi

# --- 2. Configure Git ---
echo -e "\n⚙️  Configuring Git global settings..."
git config --global user.name "$GITHUB_NAME"
git config --global user.email "$GITHUB_EMAIL"
echo "✅ Git configured."

# --- 3. Clone Project Repository ---
echo -e "\n⬇️  Cloning project repository..."
git clone https://github.com/benji-benji/mlx_week5_audio
cd mlx_week5_audio
echo "✅ Cloned and changed directory to $(pwd)"

# --- 4. Install System Dependencies ---
echo -e "\n📦 Installing system dependencies with apt..."

apt update
apt install -y vim rsync git git-lfs nvtop htop tmux curl btop
echo "✅ System dependencies installed."

# --- 5. Install and Initialize Miniconda ---
if ! command -v conda &> /dev/null
then
    echo -e "\n🐍 Installing Miniconda..."
    # Download the installer
    curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    # Run the installer in batch mode (non-interactive)
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
    # Clean up the installer
    rm Miniconda3-latest-Linux-x86_64.sh
    echo "✅ Miniconda installed."
else
    echo -e "\n🐍 Miniconda is already installed."
fi

# Initialize conda for the current shell session
echo "🚀 Initializing Conda for this script session..."
eval "$($HOME/miniconda/bin/conda shell.bash hook)"
# Permanently initialize for future interactive sessions
conda init bash
echo "✅ Conda initialized."

# --- 6. Create and Activate Conda Environment ---
echo -e "\n🌿 Creating conda environment from environment.yml..."
conda update -n base conda
conda env create -f environment.yml
echo "(Activating) conda environment..."
conda activate audio_classifier
echo "✅ Conda environment 'audio_classifier' is active."

# --- 7. Final Checks and Launch Tmux ---
echo -e "\n🔍 Final checks..."
echo -n "Python path: "
which python
echo -n "Conda env: "
echo $CONDA_DEFAULT_ENV

echo -e "\n🎉 Setup complete! You are inside the 'audio_classifier' environment."
echo "🚀 Launching tmux session..."

tmux