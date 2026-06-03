#!bin/bash

cd ~/data/cubicasa5k/;

curl -L -o cubicasa5k.zip \
"https://zenodo.org/records/2613548/files/cubicasa5k.zip?download=1";

unzip cubicasa5k.zip;

cd cubicasa5k/cubicasa5k/

python create_cubicasa5k_walls.py -d c
python create_cubicasa5k_walls.py -d hq
python create_cubicasa5k_walls.py -d hqa

cp -r cubicasa5k ~/dev/arch-segmentation-training/data/raw 