# Data Process
This folder contains the scripts to process the data for training and inference.

## Installation
Install the torch and torchvision packages:
```bash
conda create -n waymo python=3.8
pip install torch==1.8.1 torchvision==0.9.1
pip install -r requirements.txt
```
## Training Data

### Download and Preprocess Waymo Dataset
To begin, download the data from the [Waymo Open Dataset](https://waymo.com/open/). Subsequently, convert the data into the required format using the provided `preprocess.py` script. 

For instance, to convert `segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord`, execute the following command:
```bash
conda activate waymo
cd data_process
python preprocess.py --root /path/to/tfrecord --out_root /path/to/tfrecord_processed -j4 --seq_list=need_process.lst
```

Each `.tfrecord` file contains point cloud data and camera data, organized as shown below:
```
./segment-1005081002024129653_5313_150_5333_150_with_camera_labels
├── images
│   ├── camera_FRONT
│   │   ├── 00000000.jpg
│   │   ├── 00000001.jpg
│   │   └── ...
│   ├── camera_FRONT_LEFT
│   ├── camera_FRONT_RIGHT
│   ├── camera_SIDE_LEFT
│   └── camera_SIDE_RIGHT
├── lidars
│   ├── lidar_TOP
│   │   ├── 00000000.npz
│   │   └── ...
│   ├── lidar_FRONT
│   ├── lidar_REAR
│   ├── lidar_SIDE_LEFT
│   └── lidar_SIDE_RIGHT
└── scenario.pt
```
The `scenario.pt` file is a dictionary that stores scene information such as camera intrinsic and extrinsic parameters, along with bounding box annotations.

### Semantic Segmentation GT Generation
To generate semantic segmentation ground truth, we utilize SegFormer. Follow the instructions for installing the environment as [SegFormer](https://github.com/NVlabs/SegFormer), using the `segformer.b3.1024x1024.city.160k.pth` model. 

Execute the following command to generate the semantic segmentation ground truth:
```bash
cd data_process

# Install SegFormer
conda activate waymo
pip install timm==0.4.12
pip install mmcv-full==1.2.7 --no-cache-dir

cd third_party/SegFormer
pip install -e . --user

cd ../../
python pipeline_segment.py --data_root /path/to/data/segment-1005081002024129653_5313_150_5333_150_with_camera_labels
```

The generated semantic maps will be saved in the `segment-1005081002024129653_5313_150_5333_150_with_camera_labels` folder as follows:
```
./segment-1005081002024129653_5313_150_5333_150_with_camera_labels
├── semantic
│   ├── camera_FRONT
│   │   ├── seg
│   │   │   ├── 00000000.pfm
│   │   │   └──...
├── camera_FRONT_LEFT
├── camera_FRONT_RIGHT
├──...
```

### Sparse Conditional Generation
We first aggregate point clouds and project them into camera views to obtain sparse depth maps and colorized semantic segmentation results. 
The script `pipeline_process.py` is provided for this process.
```bash
conda activate waymo
python pipeline_process.py --data_path /path/to/data/segment-1005081002024129653_5313_150_5333_150_with_camera_labels
```
Additionally, in this section, we visualize the semantic maps in the `semantic_dense` folder.

### Depth GT Generation
Aggregate 6-frame lidar data using `pipeline_depth.py`:
```bash
conda activate waymo
python pipeline_depth.py --data_path /path/to/data/segment-1005081002024129653_5313_150_5333_150_with_camera_labels
```

Then, generate dense depth maps using DepthLab. Follow the instructions to install the environment as [DepthLab](https://github.com/ant-research/DepthLab). 
Subsequently, copy `depthlab_tools.py` to DepthLab's root directory and execute:
```bash
cp depthlab_tools.py ./third_party/DepthLab

cd ./third_party/DepthLab
conda activate DepthLab
python depthlab_tools.py --data_path /path/to/data/segment-1005081002024129653_5313_150_5333_150_with_camera_labels
```

### Data Structure
In the end, the processed data will be saved in the following structure:
```
./segment-1005081002024129653_5313_150_5333_150_with_camera_labels
├── images                       # camera images
├── lidars                       # lidar data
├── semantic                     # semantic segmentation
├── semantic_dense               # colorized semantic segmentation
├── depth                        # sparse depth data
├── sparse                       # sparse RGB data
├── objects                      # object point cloud data
├── objects_info.pkl             # object information
├── six_frames_depth             # aggregated 6-frame lidar data
├── six_frames_depth_dense       # dense depth data from DepthLab
├── six_frames_depth_aligned     # aligned six_frames_depth_dense data with sparse RGB data
├── six_frames_depth_processed   # processed six_frames_depth_aligned data
├── six_frames_depth_vis         # visualization of depth data
├── six_frames_sparse            # aggregated 6-frame sparse RGB data
└── scenario.pt                  # scene information
```


## Inference Data
### Inference Example
We provide a pre-processed example dataset available for direct inference at:  
[https://huggingface.co/datasets/heiheishuang/MuDG_waymo_example/](https://huggingface.co/datasets/heiheishuang/MuDG_waymo_example/)

### Generate Virtual Condition
Use `pipeline_process.py` to generate novel-view conditions:
```bash
conda activate waymo
python pipeline_process.py --data_path /path/to/data/segment-1005081002024129653_5313_150_5333_150_with_camera_labels --render_virtual
```
Novel-view sparse RGB and depth conditions will be saved in `virtual_sparse` and `virtual_depth` folders respectively:
```
./segment-1005081002024129653_5313_150_5333_150_with_camera_labels
├── virtual_sparse
│   ├── camera_FRONT
│   │   ├── 00000001_1.jpg // Rendered novel-view condition (left-2m)
│   │   ├── 00000001_1_bg.jpg // Background component
│   │   ├── 00000001_1_mask.jpg // Mask
│   │   ├── 00000001_1_obj.jpg // Object-only component
│   │   └── ...
├── virtual_depth
│   ├── camera_FRONT
│   │   ├── 00000001_1.npy // Novel-view depth map
│   │   ├── 00000001_1_bg.npy // Background depth
│   │   ├── 00000001_1_obj.npy // Object depth
│   │   └── ...
├── ...
```

## Acknowledgements
We would like to thank the contributors of the following repositories for their valuable contributions to the community:
- [neuralsim](https://github.com/PJLab-ADG/neuralsim)
- [SegFormer](https://github.com/NVlabs/SegFormer)
- [DepthLab](https://github.com/ant-research/DepthLab)