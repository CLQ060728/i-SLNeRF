export CUDA_HOME=/usr/local/cuda-12.1/
printf "CUDA_HOME set to: %s\n" "$CUDA_HOME"

git clone https://github.com/xinyu1205/recognize-anything.git ./RAMPP/
rm -rf ./RAMPP/.git
rm -rf ./RAMPP/.github
rm -rf ./RAMPP/.gitignore
rm -rf ./RAMPP/.gitmodules
printf "Cloned RAM++ repository and removed unnecessary git files\n"
cp ./inference_ram_plus.py ./RAMPP/
cp ./inference_ram_plus_openset.py ./RAMPP/
cp ./rampp_requirements.txt ./RAMPP/requirements.txt

cd ./RAMPP/
pip install -r requirements.txt
pip install --no-build-isolation -e .

mkdir ./pretrained/
wget -P ./pretrained/ https://huggingface.co/xinyu1205/recognize-anything-plus-model/blob/main/ram_plus_swin_large_14m.pth
