# Download VG images
if [ ! -d "VG" ]; then
    mkdir VG
    cd VG
    wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
    wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip
    unzip images.zip
    unzip images2.zip
    rm images.zip images2.zip
    mv VG_100K_2/* VG_100K/
    rm -r VG_100K_2/
    mv VG_100K/ images/
    cd ..
fi

# Download VRD images
if [ ! -d "VRD" ]; then
    wget http://imagenet.stanford.edu/internal/jcjohns/scene_graphs/sg_dataset.zip
    unzip sg_dataset.zip
    rm sg_dataset.zip
    mv sg_dataset/sg_test_images/* sg_dataset/sg_train_images/
    rm -r sg_dataset/sg_test_images/
    rm sg_dataset/sg_test_annotations.json
    rm sg_dataset/sg_train_annotations.json
    mv sg_dataset/sg_train_images/ sg_dataset/images/
    mv sg_dataset/ VRD/
fi

# Download UnRel images
if [ ! -d "UnRel" ]; then
    mkdir UnRel
    cd UnRel
    wget http://www.di.ens.fr/willow/research/unrel/data/unrel-dataset.tar.gz
    tar -xvf unrel-dataset.tar.gz
    rm unrel-dataset.tar.gz
    rm annotations.mat
    rm annotatated_triplets.mat
    cd ..
fi