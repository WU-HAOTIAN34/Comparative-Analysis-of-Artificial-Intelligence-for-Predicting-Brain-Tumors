# Comparative Analysis of Artificial Intelligence for Predicting Brain Tumors

## 1. Environment Configeration

```
conda init
conda create --name tumor -y python=3.8
conda activate tumor
python -m pip install --upgrade pip
```

```
sudo apt-get install unzip
unzip .zip
```

```
pip install keras
pip install scikit-learn
pip install seaborn
pip install opencv-python
pip install tqdm
pip install tensorflow==2.9.0
pip install ipykernel
conda list tensorflow
```

## 2. Dataset and Preprocessing

There are four dataset:  
[Br35H](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection?select=no)  
[SARTAJ's dataset](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)  
A widely used public multiclassification brain tumor dataset ([MBTD](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427))   
[Br35H+SARTAJ+figshare](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)  

```
pip install hdf5storage
pip install split-folders
pip install imutils
```

Delete all pred0.jpg file in Br35H+SARTAJ+figshare, SARTAJ_dataset and MBTD.

```
find /path/to/directory -name 'rew.png' -type f -delete
```

To use the dataset:
1. Split training set and testing set by Split_folders.ipynb
2. Crop brain tumor by Crop_Brain_Contours.ipynb
3. Augmentation by Data_Augmentation.ipynb 

Download MBTD you can:
```
# create a directory
mkdir brain_tumor_dataset
cd brain_tumor_dataset

# download the dataset
wget https://ndownloader.figshare.com/articles/1512427/versions/5

# unzip the dataset and delete the zip
unzip 5 && rm 5

# concatenate the multiple zipped data in a single zip
cat brainTumorDataPublic_* > brainTumorDataPublic_temp.zip
zip -FF brainTumorDataPublic_temp.zip --out data.zip

# remove the temporary files
rm brainTumorDataPublic_*

# unzip the full archive and delete it 
unzip data.zip -d data && rm data.zip

# check that "data" contains 3064 files
ls data | wc -l
```

Use visualize.ipynb to extract images from .mat file 
