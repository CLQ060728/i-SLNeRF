git clone https://github.com/facebookresearch/sam2.git && cd sam2
rm -rf ./.git
pip install -e .
mkdir ./checkpoints/
wget -P ./checkpoints/ https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
