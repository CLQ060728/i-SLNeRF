export CUDA_HOME=/usr/local/cuda-12.1/
printf "CUDA_HOME set to: %s\n" "$CUDA_HOME"

git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
rm -rf ./Grounded-SAM-2/.git
rm -rf ./Grounded-SAM-2/.github
rm -rf ./Grounded-SAM-2/.gitignore
rm -rf ./Grounded-SAM-2/.gitmodules
rm -rf ./Grounded-SAM-2/grounding_dino/requirements.txt
cp ./gdino_requirements.txt ./Grounded-SAM-2/grounding_dino/requirements.txt
printf "Removed unnecessary git files and copied requirements.txt\n"
bash ./Grounded-SAM-2/gdino_checkpoints/download_ckpts.sh
cd Grounded-SAM-2
printf "Changed directory to Grounded-SAM-2\n"
pip install --no-build-isolation -e grounding_dino

