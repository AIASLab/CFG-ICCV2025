## Environment
This codebase was tested with the following environment configurations.

- Debian GNU/Linux 12
- GCC 12.2.0
- CUDA 11.3
- Python 3.7
- PyTorch 1.12.0


## Installation

* Download this git repository and install the requirement from the root directory. 
```
git clone 
```

* Install the other dependencies:
```

# Install PointNet++
cd Model/pointnet2_ops_lib/
python setup.py install

# Install pytorch3d>=0.7.1
https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md
```

## Data Preparation and Pretrained Model

#### ModelNet40

* Download the [ModelNet40](https://drive.google.com/file/d/10UC5IfUW9fJTjJX3hyo-1yrsYfKuDQEL/view) dataset.

* Prepare the dataset paths as follows: ```./data/data.Mat```

#### Pretrained Model
* Download [Pretrained Model](https://drive.google.com/drive/folders/1VdYDhOmitsZYRLc4V04GCm2f0Atb3lt3?usp=sharing).
* Uncompressed the Pretrained Model to folder ```./Pretrained/```. We provided pretrained models of ```PointNet, PointNetPP_msg, PointNetPP_ssg, DGCNN, pointconv,  pointcnn, curvenet, PCT, Point-pn```. The paths are organized as follows:
```bash
.
├── Pretrained
    ├── PointNet
    │   └── 1024
    │       ├── checkpoint.pth.tar
    │       ├── model_best.pth.tar
    │       └── result.txt
    ├── DGCNN
    │   └── 1024
    │       ├── checkpoint.pth.tar
    │       ├── model_best.pth.tar
    │       └── result.txt
    ├── pointconv
    │   └── 1024
    │       ├── checkpoint.pth.tar
    │       ├── model_best.pth.tar
    │       └── result.txt
    ├── PCT   

```

## Attack 

To generate the adversarial point clouds using our attacker with the source model, 
```
# set SOURCE_MODEL = ''
python attack.py --source_model=SOURCE_MODEL
```
The default source model is : ```PointNet```. 

The generated adversarial point clouds are saved to the directory of ```./Results/ ```


## Evaluation Transferability


To evaluate the transferability of the adversarial point clouds.   
```
# set TARGET_MODEL = '' , 
# This codebase supports the following target models: 
# [PointNet, PointNetPP_msg, PointNetPP_ssg, DGCNN, pointconv,  pointcnn, curvenet, PCT, Point-pn]

python evaluate.py --target_model=$TARGET_MODEL --datadir=path/to/adverarial_point_clouds
```


## Defense

Verifying the performance of generated adversarial point clouds in defense scenarios,

For defense methods of CCN and AT,run 
```
# set TARGET_MODEL='ccn' or 'PointNet-AT'
python evaluate.py --target_model=$TARGET_MODEL  --datadir=path/to/adverarial_point_clouds
```
For other defense methods, run 
```
# set DEFENSE_TYPE = '' 
python defense.py --target_model=PointNet --datadir=path/to/adverarial_point_clouds --defense_type=$DEFENSE_TYPE
```

## To be released
Pretrained models on ScanObjectNN dataset will be released later.


