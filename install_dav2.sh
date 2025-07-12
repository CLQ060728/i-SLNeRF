export CUDA_HOME=/usr/local/cuda-12.1/
printf "CUDA_HOME set to: %s\n" "$CUDA_HOME"

git clone https://github.com/DepthAnything/Depth-Anything-V2.git ./DAV2/
rm -rf ./DAV2/.git
rm -rf ./DAV2/.github
rm -rf ./DAV2/.gitignore
rm -rf ./DAV2/.gitmodules
printf "Cloned DAV2 repository and removed unnecessary git files\n"

cp ./dav2_requirements.txt ./DAV2/requirements.txt
cp ./inference_dav2.py ./DAV2/inference_dav2.py

cd ./DAV2/
printf "Changed directory to DAV2\n"
pip install --upgrade pip==25.0
pip install -r requirements.txt
mkdir checkpoints
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true -O checkpoints/depth_anything_v2_vitl.pth

pip cache purge
