# rm -rf ./Grounded-SAM-2/sam2/ 
git clone https://github.com/facebookresearch/sam2.git ./Grounded-SAM-2/SAM2/
rm -rf ./Grounded-SAM-2/SAM2/.git
rm -rf ./Grounded-SAM-2/SAM2/.github
rm -rf ./Grounded-SAM-2/SAM2/.gitignore
rm -rf ./Grounded-SAM-2/SAM2/.gitmodules
printf "Removed unnecessary git files from SAM2\n"
cd ./Grounded-SAM-2/
pip install --no-build-isolation -e SAM2
mkdir ./checkpoints/
wget -P ./checkpoints/ https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
cp ./checkpoints/sam2.1_hiera_large.pt ./SAM2/checkpoints/
cp ../sam2_feature_extractor.py ./SAM2/
rm -rf ./sam2/
cp -r ./SAM2/sam2/ ./
export CUDA_HOME=/usr/local/cuda-12.1/
pip install --upgrade torch==2.5.1
pip install --upgrade torchvision==0.20.1
pip install --upgrade xformers==0.0.28.post3
pip install --upgrade tqdm==4.67.1
pip install --upgrade pillow==11.2.1
pip install --upgrade numpy==1.26.4
pip install --upgrade scipy==1.15.3
pip install --upgrade pandas==2.3.0
pip install --upgrade scikit-learn==1.7.0
pip install --upgrade matplotlib==3.10.3
pip install --upgrade plotly==6.1.2
pip install --upgrade seaborn==0.13.2
pip install --upgrade opencv-python==4.11.0.86

pip cache purge