#!/bin/bash

mkdir -p $1

# Download our YouTube8M dataset
python dataset/download_process_youtube_8m.py --resolution 256 --fps 4 --output_folder $1/youtube_8m_dataset_r256

# Process LOVU-TGVE
gdown 1D7ZVm66IwlKhS6UINoDgFiFJp_mLIQ0W -O $1/loveu-tgve-2023.zip
sleep .5
unzip $1/loveu-tgve-2023.zip -d $1
rm $1/loveu-tgve-2023.zip
python dataset/process_loveu.py --resolution 256 --fps 4 --output_folder $1/loveu_dataset_r256 --loveu_folder $1/loveu-tgve-2023

# Dreamix Dataset
python dataset/download_process_dreamix_dataset.py --resolution 256 --fps 4 --output_folder $1/dreamix_dataset_r256