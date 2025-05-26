<p align="center">
    <picture>
    <img alt="pointcept" src="https://dhl3d.github.io/images/PointSegBase_badge.jpg">
    </picture><br>
</p>

PointSegBase is a code base for point cloud segmentation. It is a comprehensive and easy-to-use code base for point cloud segmentation, including various models, datasets, and evaluation metrics.  
[![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2FPointCloud-Lab%2FPointSegBase%3Ftab%3Dreadme-ov-file&label=Visitors&countColor=%23fedcba&style=flat&labelStyle=none)](https://visitorbadge.io/status?path=https%3A%2F%2Fgithub.com%2FPointCloud-Lab%2FPointSegBase%3Ftab%3Dreadme-ov-file)  
## NEWS

- [2025/05/25] Basic code is released.
- [2025/02/05] 🚀 Our work 'Deep Hierarchical Learning for 3D Semantic Segmentation' is accepted by IJCV.
- [2024/08/12] 🚀 Our work 'Pyramid Diffusion for Fine 3D Large Scene Generation' is selected as **Oral Presentation**.
- [2024/07/02] 🚀 Our work 'Pyramid Diffusion for Fine 3D Large Scene Generation' is accepted by ECCV 2024.
- [2023/11/29] Official repo is created, code will be released soon, access our [Project Page](https://dhl3d.github.io/) for more details.

## Related Projects

There are several works that are based / used PointSegBase, including:

- **📌 Deep Hierarchical Learning for 3D Semantic Segmentation**  
*Chongshou Li, Yuheng Liu, Xinke Li, Yuning Zhang, Tianrui Li, Junsong Yuan*  
International Journal of Computer Vision (**IJCV**) 2025  
[ [Project](https://dhl3d.github.io/) ] [ [Paper](https://link.springer.com/article/10.1007/s11263-025-02387-6) ]


- **Pyramid Diffusion for Fine 3D Large Scene Generation**  
*Yuheng Liu, Xinke Li, Xueting Li, Lu Qi, Chongshou Li, Ming-Hsuan Yang*  
European Conference on Computer Vision (**ECCV**) 2024 - **Oral Presentation (top 2%)**  
[ [Project](https://yuheng.ink/project-page/pyramid-discrete-diffusion/) ] [ [arXiv](https://arxiv.org/abs/2311.12085) ] [ [Code](https://github.com/yuhengliu02/pyramid-discrete-diffusion) ]

## Citations
If you find this code base useful, please consider citing:

```bibtex
    @article{li2025deephierarchicallearningfor3dsemanticsegmentation,
        title={Deep Hierarchical Learning for 3D Semantic Segmentation},
        author={Chongshou Li and Yuheng Liu and Xinke Li and Yuning Zhang and Tianrui Li and Junsong Yuan},
        booktitle={International Journal of Computer Vision},
        year={2025}
    }
```

```bibtex
    @InProceedings{liu2024pyramiddiffusionfine3d,
        title={Pyramid Diffusion for Fine 3D Large Scene Generation},
        author={Yuheng Liu and Xinke Li and Xueting Li and Lu Qi and Chongshou Li and Ming-Hsuan Yang},
        booktitle={European Conference on Computer Vision (ECCV)},
        year={2024},
    }
```

## Installation

```bash
# Installation (Ubuntu 22.04 + CUDA 11.8)

# 1. Create a new conda environment
conda create -n PointSegBase python=3.10 -y
conda activate PointSegBase

# 2. Install PyTorch (CUDA 11.8, version < 2.0.0)
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1

# 3. Install required packages
conda install -y pyyaml six numba h5py
pip install numpy==1.25.2 scipy==1.13.0 open3d

# 4. Install TorchSparse (choose one method)

# Option A: Install via official script
python -c "$(curl -fsSL https://raw.githubusercontent.com/mit-han-lab/torchsparse/master/install.py)"

# Option B: Install directly from GitHub
sudo apt-get install libsparsehash-dev
pip install git+https://github.com/mit-han-lab/torchsparse.git
```

## Datasets
We provide the following datasets:

- **Campus3D, Urban3D, PartNet-H**  
[ [Hugging Face](https://huggingface.co/datasets/Yuheng02/PointSegBase_Datasets) ] [ [Baidu Netdisk](https://pan.baidu.com/s/1pUnCJXRICnGuA_EWG8QHIA?pwd=2025) (Until 05/25/2026) ]  

All datasets have been preprocessed and should be placed in the `data` folder.

## Training
Before starting any training, you need to set the corresponding config file located in the `configs` folder. Use `-c` to specify the config path for the current experiment group, and `--exp_name` to set the path where logs and checkpoints for the current experiment group will be saved. We provide example configs for both single-layer and multi-layer training on the **Campus3D** and **Urban3D** datasets using `UNet` and `PointNet++` networks, as well as for the `Bed` category in the **PartNet-H** dataset using the same networks. You can refer to these examples to create additional training configs for other categories in PartNet or for other network architectures.

```bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
python run.py -c ${CONFIG_PATH} --exp_name ${EXP_NAME}
```

### Campus3D
```bash
# Settings (Campus3D, PointNet++, layer 1)
python run.py -c configs/pointnet_campus_layer_1 --exp_name exp_campus3d_pointnet_layer_1

# Settings (Campus3D, PointNet++, layer 2)
python run.py -c configs/pointnet_campus_layer_2 --exp_name exp_campus3d_pointnet_layer_2

# Settings (Campus3D, PointNet++, layer 3)
python run.py -c configs/pointnet_campus_layer_3 --exp_name exp_campus3d_pointnet_layer_3

# Settings (Campus3D, PointNet++, layer all)
python run.py -c configs/pointnet_campus_layer_all --exp_name exp_campus3d_pointnet_layer_all

# Settings (Campus3D, UNet, layer 1)
python run.py -c configs/unet_campus_layer_1 --exp_name exp_campus3d_unet_layer_1

# Settings (Campus3D, UNet, layer 2)
python run.py -c configs/unet_campus_layer_2 --exp_name exp_campus3d_unet_layer_2

# Settings (Campus3D, UNet, layer all)
python run.py -c configs/unet_campus_layer_all --exp_name exp_campus3d_unet_layer_all
```

### Urban3D
```bash
# Settings (Urban3D, PointNet++, layer 1)
python run.py -c configs/pointnet_urban_layer_1 --exp_name exp_urban3d_pointnet_layer_1

# Settings (Urban3D, PointNet++, layer 2)
python run.py -c configs/pointnet_urban_layer_2 --exp_name exp_urban3d_pointnet_layer_2

# Settings (Urban3D, PointNet++, layer all)
python run.py -c configs/pointnet_urban_layer_all --exp_name exp_urban3d_pointnet_layer_all

# Settings (Urban3D, UNet, layer 1)
python run.py -c configs/unet_urban_layer_1 --exp_name exp_urban3d_unet_layer_1

# Settings (Urban3D, UNet, layer 2)
python run.py -c configs/unet_urban_layer_2 --exp_name exp_urban3d_unet_layer_2

# Settings (Urban3D, UNet, layer all)
python run.py -c configs/unet_urban_layer_all --exp_name exp_urban3d_unet_layer_all
```

### PartNet-H
```bash
# Settings (PartNet-H-Bed, PointNet++, layer 1)
python run.py -c configs/pointnet_partnet-Bed_layer_1 --exp_name exp_partnet-Bed_pointnet_layer_1

# Settings (PartNet-H-Bed, PointNet++, layer 2)
python run.py -c configs/pointnet_partnet-Bed_layer_2 --exp_name exp_partnet-Bed_pointnet_layer_2

# Settings (PartNet-H-Bed, PointNet++, layer 3)
python run.py -c configs/pointnet_partnet-Bed_layer_3 --exp_name exp_partnet-Bed_pointnet_layer_3

# Settings (PartNet-H-Bed, PointNet++, layer all)
python run.py -c configs/pointnet_partnet-Bed_layer_all --exp_name exp_partnet-Bed_pointnet_layer_all

# Settings (PartNet-H-Bed, UNet, layer 1)
python run.py -c configs/unet_partnet-Bed_layer_1 --exp_name exp_partnet-Bed_unet_layer_1

# Settings (PartNet-H-Bed, UNet, layer 2)
python run.py -c configs/unet_partnet-Bed_layer_2 --exp_name exp_partnet-Bed_unet_layer_2

# Settings (PartNet-H-Bed, UNet, layer 3)
python run.py -c configs/unet_partnet-Bed_layer_3 --exp_name exp_partnet-Bed_unet_layer_3

# Settings (PartNet-H-Bed, UNet, layer all)
python run.py -c configs/unet_partnet-Bed_layer_all --exp_name exp_partnet-Bed_unet_layer_all
```

## Evaluation
After completing the training of your model, you can use `--eval` to evaluate the trained model. Specifically, you need to go into the folder corresponding to the experiment name you previously set, and edit the `train_setting.yaml` file in the configs folder. Set `IS_PRETRAINED` to `True` and `PRETRAINED_MODEL_PATH` to the path of the model inside the checkpoints folder.

```bash
python run.py -c ${EXP_NAME}/configs --exp_name ${EVAL_NAME} --eval
```

### Example

```bash
python run.py -c exp_campus3d_pointnet_layer_1/configs --exp_name exp_campus3d_pointnet_layer_1_eval --eval
```


## Contact

This repository will be maintained by [Yuheng](https://yuheng.ink). For any issues regarding code, dataset downloads, or the official website, you can contact Yuheng at the following email addresses: yuhengliu02@gmail.com.
