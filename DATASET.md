# Dataset

### 3D-Points Pre-training

For using our pre-training data (76,014 episodes), please follow the Dataset Setup instructions outlined [here](https://github.com/yuvansharma/SpaTracker?tab=readme-ov-file#1-dataset-setup).

After setting up the dataset, make sure to update [dataset_config.json](config/dataset_config_epic_pretraining.json) so that the file points to the correct location. The ```dataset_dir``` field should be set to the root folder which contains the ```epic_tasks_final``` folder downloaded from HuggingFace.

### Robotics Fine-tuning
#### Structure
The structure of the data using in the fine-tuning phase is shown as follows:

```
anns/
│ 
└── task1
│   ├── kinova_tasks/common_task/eps1
│   │   ├── instruction.txt 
│   │   │   └── (instruction of the eps using natural language)
│   │   ├── proprio.zarr 
│   │   │   └── (array of data with size T * 7 (x, y, z, yaw, roll, pitch, gripper))
│   │   ├── instruction.zarr 
│   │   │   └── (array of text embedding with size (768,))
│   │   └── action.zarr
│   │   │   └── (array of data with size T * 7 (x, y, z, yaw, roll, pitch, gripper))
│   │   └── images
│   │       └── exterior_image_1_left
│   │           └── (images saved as xxxxx.jpg)
│   │       └── wrist_image_left
│   │           └── (images saved as xxxxx.jpg)
│   │   
│   └── kinova_tasks/common_task/eps2
│       ...
│
└── task2
    ... 
```

#### Download our Kinova Dataset
We release the real multi-task data used in our experiments with the Kinova Gen3 robot arm. You can download it on [🤗HuggingFace](https://huggingface.co/datasets/yuvansharma/arm4r-data). 
Download the ```real_kinova_release_data.zip``` file and update [dataset_config.json](config/dataset_config_kinova.json) so that the file points to the correct location.
```bash 
# install git-lfs
sudo apt install git-lfs
git lfs install
# clone the dataset
git clone https://huggingface.co/datasets/yuvansharma/arm4r-data
# or you can download the files manually from here: https://huggingface.co/datasets/yuvansharma/arm4r-data
```


## Use Your Own Dataset
You can convert your own data to the above format to train or fine-tune your own model. 
We provide brief examples for this conversion for RLBench sim data to our format. For details, see [SIM.md](SIM.md).