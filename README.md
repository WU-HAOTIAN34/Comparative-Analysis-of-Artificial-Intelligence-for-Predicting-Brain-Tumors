# Brain-Tumor

## 1.

```
conda create --name nerfstudio -y python=3.8
conda activate nerfstudio
python -m pip install --upgrade pip
```

```
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

```
python -c 'import torch;print(torch.backends.cudnn.version())'
python -c 'import torch;print(torch.__version__)'
```

```
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .
```

```
pip install opendatasets
python
import opendatasets as od
od.download("https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1")
```

```
```

```
```

```
```

```
```
