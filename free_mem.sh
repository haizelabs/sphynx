#!/bin/bash

echo "Attempting to reset GPU memory..."

# Find and kill processes using the GPU
gpu_pids=$(fuser -v /dev/nvidia* 2>/dev/null | awk '{print $2}')
if [ -n "$gpu_pids" ]; then
    echo "Killing the following GPU processes: $gpu_pids"
    sudo kill -9 $gpu_pids
else
    echo "No GPU processes found."
fi

# Unload NVIDIA kernel modules if not in use
sudo rmmod nvidia_uvm 2>/dev/null || echo "Failed to unload nvidia_uvm"
sudo rmmod nvidia_modeset 2>/dev/null || echo "Failed to unload nvidia_modeset"
sudo rmmod nvidia 2>/dev/null || echo "Failed to unload nvidia"

# Reload NVIDIA kernel modules
sudo modprobe nvidia
sudo modprobe nvidia_uvm
sudo modprobe nvidia_modeset

# Check if nvidia-persistenced service exists and restart it if available
if systemctl list-units --full -all | grep -q 'nvidia-persistenced.service'; then
    sudo systemctl restart nvidia-persistenced
else
    echo "nvidia-persistenced service not found."
fi

echo "GPU memory should now be freed up."
