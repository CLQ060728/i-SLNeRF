pip install --upgrade -r requirements.txt
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install git+https://github.com/nerfstudio-project/nerfacc.git@8340e19daad4bafe24125150a8c56161838086fa
pip install git+https://github.com/facebookresearch/sam2.git
mkdir -p ./checkpoints/
wget -P ./checkpoints/ https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
pip cache purge