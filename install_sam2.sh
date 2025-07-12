git clone https://github.com/facebookresearch/sam2.git && cd sam2
rm -rf ./.git
pip install -e .
mkdir ./checkpoints/
wget -P ./checkpoints/ https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
cp ../sam2_feature_extractor.py ./
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