export CUDA_HOME=/usr/local/cuda-12.1/
printf "CUDA_HOME set to: %s\n" "$CUDA_HOME"

git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
rm -rf ./Grounded-SAM-2/.git
rm -rf ./Grounded-SAM-2/.github
rm -rf ./Grounded-SAM-2/.gitignore
rm -rf ./Grounded-SAM-2/.gitmodules
printf "Removed unnecessary git files\n"
bash ./Grounded-SAM-2/gdino_checkpoints/download_ckpts.sh
mv ./groundingdino_swinb_cogcoor.pth ./Grounded-SAM-2/gdino_checkpoints/
mv ./groundingdino_swint_ogc.pth ./Grounded-SAM-2/gdino_checkpoints/
git clone https://github.com/IDEA-Research/Grounding-DINO-1.5-API.git ./Grounded-SAM-2/grounding_dino/Grounding-DINO-1.5-API/
rm -rf ./Grounded-SAM-2/grounding_dino/Grounding-DINO-1.5-API/.git
rm -rf ./Grounded-SAM-2/grounding_dino/Grounding-DINO-1.5-API/.github
rm -rf ./Grounded-SAM-2/grounding_dino/Grounding-DINO-1.5-API/.gitignore
rm -rf ./Grounded-SAM-2/grounding_dino/Grounding-DINO-1.5-API/.gitmodules
printf "Cloned Grounding-DINO-1.5-API and removed unnecessary git files\n"
rm -rf ./Grounded-SAM-2/grounding_dino/Grounding-DINO-1.5-API/requirements.txt
rm -rf ./Grounded-SAM-2/grounding_dino/requirements.txt
cp ./gdino_requirements.txt ./Grounded-SAM-2/grounding_dino/Grounding-DINO-1.5-API/requirements.txt
cp ./gdino_requirements.txt ./Grounded-SAM-2/grounding_dino/requirements.txt
cp ./grounded_sam2_batch.py ./Grounded-SAM-2/
cp ./grounded_sam2_local_demo.py ./Grounded-SAM-2/

cd ./Grounded-SAM-2/grounding_dino/
printf "Changed directory to grounding_dino\n"
pip install --upgrade pip==24.2
pip install --upgrade setuptools==67.6.0
pip install --upgrade wheel==0.45.1    # important for compiling Grounding-DINO-1.5-API
sudo apt install g++-9 gcc-9 -y
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 100 --slave /usr/bin/g++ g++ /usr/bin/g++-9 --slave /usr/bin/gcov gcov /usr/bin/gcov-9
pip install -r requirements.txt
pip install --no-build-isolation -e .
pip install -r Grounding-DINO-1.5-API/requirements.txt
pip install --no-build-isolation -e Grounding-DINO-1.5-API

pip cache purge