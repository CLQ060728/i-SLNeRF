pip install --default-timeout=1000 --upgrade -r requirements.txt
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install git+https://github.com/nerfstudio-project/nerfacc.git@8340e19daad4bafe24125150a8c56161838086fa
pip cache purge