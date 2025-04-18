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
pip install -r requirements.txt
```

or

```
pip install keras
pip install scikit-learn
pip install seaborn
pip install opencv-python
pip install tqdm
pip install tensorflow==2.9.0
pip install keras_application
(pip install Keras-Applications)
pip install flask
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
find /path/to/directory -name 'pred0.jpg' -type f -delete
```

To use the dataset:
1. Run visualize.ipynb in dataset/figshare_MBTD
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

## 3. Implement VGG19-TLFT

```
sudo apt-get update

apt update && apt install -y libglu1-mesa-dev mesa-utils xterm xauth x11-xkb-utils xfonts-base xkb-data libxtst6 libxv1

export TURBOVNC_VERSION=2.2.5

export LIBJPEG_VERSION=2.0.90

wget http://aivc.ks3-cn-beijing.ksyun.com/packages/libjpeg-turbo/libjpeg-turbo-official_${LIBJPEG_VERSION}_amd64.deb

wget http://aivc.ks3-cn-beijing.ksyun.com/packages/turbovnc/turbovnc_${TURBOVNC_VERSION}_amd64.deb

dpkg -i libjpeg-turbo-official_${LIBJPEG_VERSION}_amd64.deb

dpkg -i turbovnc_${TURBOVNC_VERSION}_amd64.deb

rm -rf *.deb
```

```
# Start the VNC server
rm -rf /tmp/.X1*
```

```
USER=root /opt/TurboVNC/bin/vncserver :1 -desktop X -auth /root/.Xauthority -geometry 1920x1080 -depth 24 -rfbwait 120000 -rfbauth /root/.vnc/passwd -fp /usr/share/fonts/X11/misc/,/usr/share/fonts -rfbport 6006
```

```
# Check whether the vncserver process is started. If the VNCServer process exists, it is started
ps -ef | grep vnc
```

Download TurboVNC
Windows: http://aivc.ks3-cn-beijing.ksyun.com/packages/turbovnc/TurboVNC-2.2.5-x64.exe

SSH tunnel: open cmd: ssh -CNg -L 6006:127.0.0.1:6006 root@xxxxxxxxxxx -p xxxxxx

Open VNC, address: 127.0.0.1:6006

```
# Add environment variable
export DISPLAY=:1
```

You can use the following python code for simple verification, if the image is displayed in the local vnc client, the installation and startup process is correct
```
import numpy as np
import cv2


h = 500
w = 500
img = 255 * np.ones((h ,w , 3), dtype=np.uint8)
cv2.imshow("", img)
cv2.waitKey(0)
```
