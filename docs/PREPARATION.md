# Preparation

## Installation

<details close>
<summary>Requirements</summary>

* Linux (Windows is not officially supported)
* Python 3.7+
* PyTorch 1.8 or higher
* CUDA 10.1 or higher
* NCCL 2
* GCC 4.9 or higher
</details>

<details close>
<summary>Dependencies</summary>

* argparse
* dask
* decord
* fvcore
* hickle
* lpips
* matplotlib
* netcdf4
* numpy
* opencv-python
* packaging
* pandas
* scikit-image
* scikit-learn
* torch
* timm
* tqdm
* xarray==0.19.0
</details>

#### Clone the OceanVP repository

```shell
git clone https://github.com/zhenglab/OceanVP
```

#### Install the Python and PyTorch Environments

Install the corresponding versions of [Python](https://www.anaconda.com) and [PyTorch](https://pytorch.org), and also setup the conda environment.

```shell
conda env create -f environment.yml
conda activate OceanVP
```

#### Install the Dependency Packages
```shell
python setup.py develop
```

## Dataset Preparation

- Download the OceanVP dataset via the [Google Drive](https://drive.google.com/file/d/1AEnlUD0KRHbwEmKDlezWpwoQvEyX00jY/view?usp=drive_link) or [Baidu Netdisk](https://pan.baidu.com/s/1fu02GIgmConqov_VXEP97w?pwd=ovp1) links.

- Unzip and copy the dataset files to `$OceanVP/data` directory as following shows:

```
OceanVP
├── configs
└── data
    |── ocean
    |   ├── salinity_depth_0m
    |   ├── water_temp_depth_0m
    |   ├── water_u_depth_0m
    |   ├── water_v_depth_0m
```
