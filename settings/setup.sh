PROJECT_DIR=$(pwd)

# Pytorch
pip3 install -i https://download.pytorch.org/whl/cu121 -U torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
pip3 install -i https://download.pytorch.org/whl/cu121 -U xformers==0.0.27

# A modified gaussian splatting (+ alpha, depth, normal rendering)
cd extensions && git clone https://github.com/BaowenZ/RaDe-GS.git --recursive && cd RaDe-GS/submodules
pip3 install ./diff-gaussian-rasterization
cd ${PROJECT_DIR}

# Others
pip3 install -U gpustat
pip3 install -U -r settings/requirements.txt
sudo apt-get install -y ffmpeg

####### Alternative on EC2 CUDA Version: 13.0 ####### 
# Remove old toolkit
sudo apt remove nvidia-cuda-toolkit -y

# Install CUDA 13.0 from NVIDIA
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# Check available versions
apt-cache search cuda-toolkit

# Install CUDA 13.0 (or closest available)
sudo apt install cuda-toolkit-13-0 -y

# Set environment
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Build
pip install --no-build-isolation -e diff-gaussian-rasterization/
