#!/usr/bin/env bash

# gdrive_download
#
# script to download Google Drive files from command line
# not guaranteed to work indefinitely
# taken from Stack Overflow answer:
# http://stackoverflow.com/a/38937732/7002068
function gdrive_download {
    gURL=$1
    # match more than 26 word characters  
    ggID=$(echo "$gURL" | egrep -o '(\w|-){26,}')

    ggURL='https://drive.google.com/uc?export=download'

    curl -sc /tmp/gcokie "${ggURL}&id=${ggID}" >/dev/null  
    getcode="$(awk '/_warning_/ {print $NF}' /tmp/gcokie)"

    cmd='curl --insecure -C - -LOJb /tmp/gcokie "${ggURL}&confirm=${getcode}&id=${ggID}"'
    echo -e "Downloading from "$gURL"...\n"
    eval $cmd
}

# Create datasets folder to store annotations
if [ ! -d "datasets" ]; then
    mkdir datasets
fi
cd datasets

# Download VRD annotations (Lu et al. 2017)
if [ ! -d "VRD" ]; then
    mkdir VRD
    cd VRD
    wget http://cs.stanford.edu/people/ranjaykrishna/vrd/dataset.zip
    unzip dataset.zip
    rm dataset.zip
    rm -r __MACOSX
    mv dataset/* .
    rm -r dataset
    cd ..
fi

# Download VG200 annotations (Xu et al. 2017)
if [ ! -d "VG200" ]; then
    mkdir VG200
    cd VG200
    wget http://svl.stanford.edu/projects/scene-graph/dataset/VG-SGG.h5
    wget http://svl.stanford.edu/projects/scene-graph/dataset/VG-SGG-dicts.json
    wget http://svl.stanford.edu/projects/scene-graph/VG/image_data.json
    cd ..
fi

# Download VG80K annotations (Zhang et al. 2019)
if [ ! -d "VG80K" ]; then
    mkdir VG80K
    cd VG80K
    wget https://www.dropbox.com/s/minpyv59crdifk9/datasets.zip
    unzip datasets.zip
    mv datasets/large_scale_VRD/Visual_Genome/*.json .
    rm -r datasets datasets.zip
    cd ..
fi

# Download VG-MSDN annotations (Li et al. 2017)
if [ ! -d "VGMSDN" ]; then
    mkdir VGMSDN
    cd VGMSDN
    gdrive_download https://drive.google.com/file/d/1RtYidFZRgX1_iYPCaP2buHI1bHacjRTD/view?usp=sharing
    tar -xvf top_150_50.tgz
    rm top_150_50.tgz
    mv top_150_50_new/* .
    rm -r top_150_50_new
    cd ..
fi

# Download VG-VTE annotations (Zhang et al. 2017)
if [ ! -d "VGVTE" ]; then
    mkdir VGVTE
    cd VGVTE
    gdrive_download https://drive.google.com/open?id=1C6MDiqWQupMrPOgk4T12zWiAJAZmY1aa
    cd ..
fi

# Download VrR-VG annotations (Liang et al. 2019)
if [ ! -d "VrR-VG" ]; then
    mkdir VrR-VG
    cd VrR-VG
    gdrive_download https://drive.google.com/file/d/15i5J87FHfZ1DwPkumBaH50nfOg01Ja8C
    unzip VrR-VG.zip
    rm VrR-VG.zip
    wget http://svl.stanford.edu/projects/scene-graph/dataset/VG-SGG.h5
    wget http://svl.stanford.edu/projects/scene-graph/VG/image_data.json
    cd ..
fi

# Download sVG annotations (Dai et al. 2017)
if [ ! -d "sVG" ]; then
    mkdir sVG
    cd sVG
    gdrive_download https://drive.google.com/file/d/0B5RJWjAhdT04SXRfVHBKZ0dOTzQ/view?usp=sharing
    unzip svg.zip
    rm README.md svg.zip
    cd ..
fi

# Download UnRel annotations
if [ ! -d "UnRel" ]; then
    mkdir UnRel
    cd UnRel
    wget http://www.di.ens.fr/willow/research/unrel/data/unrel-dataset.tar.gz
    tar -xvf unrel-dataset.tar.gz
    rm unrel-dataset.tar.gz
    rm -r images
    cd ..
fi

# Back to parent path
cd ..