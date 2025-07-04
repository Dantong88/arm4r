# Pre-training Auto-regressive Robotic Models with 4D Representations


This repo contains the official implementation for *Pre-training Auto-regressive Robotic Models with 4D Representations*.

Foundation models pre-trained on massive unlabeled datasets have revolutionized natural language and computer vision, exhibiting remarkable generalization capabilities, thus highlighting the importance of pre-training. Yet, efforts in robotics have struggled to achieve similar success, limited by either the need for costly robotic annotations or the lack of representations that effectively model the physical world. In this paper, we introduce ARM4R, an Auto-regressive Robotic Model that leverages low-level 4D Representations learned from human video data to yield a better pre-trained robotic model. Specifically, we focus on utilizing 3D point tracking representations from videos derived by lifting 2D representations into 3D space via monocular depth estimation across time. These 4D representations maintain a shared geometric structure between the points and robot state representations up to a linear transformation, enabling efficient transfer learning from human video data to low-level robotic control. Our experiments show that ARM4R can transfer efficiently from human video data to robotics and consistently improves performance on tasks across various robot environments and configurations.

For further information, please contact [Dantong Niu](https://dantong88.github.io/) and [Yuvan Sharma](yuvan@berkeley.edu), or post an issue on Github!

<p align="center">
  <img src="assets/arm4r.png" width="800">
</p>

> [**Pre-training Auto-regressive Robotic Models with 4D Representations**](https://llarva24.github.io/)            
> [Dantong Niu*](https://dantong88.github.io/), [Yuvan Sharma*](https://scholar.google.com/citations?user=1_IIcds8es4C&hl=en), [Haoru Xue](https://haoruxue.github.io/), [Gicard Biamby](https://scholar.google.com/citations?user=s0Fof5IAAAAJ&hl=en), [Junyi Zhang](https://www.junyi42.com/), Ziteng Ji,
> [Trevor Darrell†](https://people.eecs.berkeley.edu/~trevor/), [Roei Herzig†](https://roeiherz.github.io/)      
> Berkeley AI Research, UC Berkeley    
> ICML 2025  
> [[Paper](https://arxiv.org/abs/2502.13142)] | [[Project Page](https://arm4r.github.io/)] | [[Checkpoints](https://huggingface.co/datasets/yuvansharma/arm4r-ckpts)] | [[Dataset](https://huggingface.co/datasets/yuvansharma/arm4r-data)]

## Installation
```bash
# create conda env
conda create -n arm4r python=3.8.1 -y
conda activate arm4r
# install torch [In general, CUDA 12.1 works well; later versions also work]
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# download repo 
git clone https://github.com/Dantong88/arm4r
cd arm4r 
pip install -e .
```
***

## Dataset
Please refer to [DATASET.md](DATASET.md) for instructions on downloading datasets or constructing your own dataset.
***


## Model Training 

### Download our Checkpoint Package
To download our checkpoint, run
```bash
cd arm4r
sudo apt install git-lfs
git lfs install
git clone https://huggingface.co/datasets/yuvansharma/arm4r-ckpts
```

Your folder structure should be:
```
└── arm4r/arm4r-ckpts
│    ├── model_ckpts
│    │   ├── ft_kinova_pick_cube  # Single Task Policy for Real Kinova Setting for "pick cube" task
│    │   │   ├── ft_kinova_pick_cube.pth
│    │   │   └── run.yaml
│    │   ├── ft_rlbench_meat_off_grill # Single Task Policy for Sim RLBench Setting for "meat off grill" task
│    │   │   ├── ft_rlbench_meat_off_grill.pth
│    │   │   └── run.yaml
│    │   └── pretrained_epic # first stage 3D point pre-training model weights, trained for 6 epochs
│    │       ├── pretrained_epic.pth
│    │       └── run.yaml
│    └── vision_encoder
│        └── cross-mae-rtx-vitb.pth # (pretrained vision encoder)
```
***

### 3D Points Pre-training
#### Launch the training
We provide the command below to launch 3D points pre-training with our released 
epic-kitchens data. Make sure you have followed the [instructions]((DATASET.md#3d-points-pre-training)) to download and format the data, and updated the path in the [dataset_config_epic_pretraining.json](config/dataset_config_epic_pretraining.json),
then launch the pre-training using:
```bash
torchrun --nproc_per_node=8 --master_port=2450 scripts/pretrain_epic.py --dataset-cfg.dataset-json config/dataset_config_epic_pretraining.json --logging-cfg.output-dir output --logging-cfg.log-name pretrain_epic --optimizer-cfg.warmup-epochs 1.25 --trainer-cfg.epochs 10 --model-cfg.vision-encoder-cfg.vision-encoder arm4r-ckpts/vision_encoder/cross-mae-rtx-vitb.pth --dataset-cfg.num-repeat-traj 1 --model-cfg.policy-cfg.no-prompt-loss --model-cfg.policy-cfg.task 3dpoints --model-cfg.policy-cfg.scratch-llama-config config/model_config/custom_transformer.json --dataset-cfg.non-overlapping 1 --shared-cfg.save-every 1 --dataset-cfg.shuffle-repeat-traj --optimizer-cfg.lr 5e-4 --shared_cfg.batch_size 64 --shared-cfg.num_pred_steps 1 --model-cfg.policy-cfg.proprio-dim 3888 --model-cfg.policy-cfg.action-dim 3889 --shared-cfg.seq_length 16
```
#### Model Inference
Run inference to get the model's 3D point tracks prediction:
```bash
python tools/inference_points_epic.py
```
It will save the predicted tracking results to the test demo folder as a `.npy` file. To visualize these results, you can refer to the instructions [here](https://github.com/yuvansharma/SpaTracker?tab=readme-ov-file#visualization).

***
### Robotic Data Fine-tuning

#### 1. SIM
We provide step-by-step instructions on how to reproduce our results on RLBench, including
converting official RLBench demonstrations to our format, training, and evaluating the arm4r model.
See the details in [SIM.md](SIM.md).

#### 2. Real on Kinova
##### Fine-tuning
We provide the command to launch fine-tuning of the robotics tasks using our released 
Kinova data. Make sure you have updated the path in the [dataset_config_kinova.json](config/dataset_config_kinova.json),
then launch the fine-tuning using:

```bash
cd arm4r
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 --master_port=2453 scripts/finetune_robotics.py --dataset-cfg.dataset-json config/dataset_config_kinova.json --logging-cfg.output-dir output --logging-cfg.log-name finetune_kinova --optimizer-cfg.warmup-epochs 1.25 --trainer-cfg.epochs 125 --model-cfg.vision-encoder-cfg.vision-encoder arm4r-ckpts/vision_encoder/cross-mae-rtx-vitb.pth --dataset-cfg.num-repeat-traj 1 --model-cfg.policy-cfg.no-prompt-loss --model-cfg.policy-cfg.task robotics --model-cfg.policy-cfg.scratch-llama-config config/model_config/custom_transformer.json --dataset-cfg.non-overlapping 1 --shared-cfg.save-every 5 --dataset-cfg.shuffle-repeat-traj --optimizer-cfg.lr 5e-4 --shared_cfg.batch_size 64 --shared-cfg.seq_length 16 --model-cfg.policy-cfg.pretrained_path arm4r-ckpts/model_ckpts/pretrained_epic/pretrained_epic.pth
```

##### Model Inference

Please look at [inference_kinova.ipynb](tools/inference_kinova.ipynb) for examples of running inference for ARM4R.

***
## License
This project is under the Apache 2.0 license. See [LICENSE](LICENSE.txt) for details. This work was supported by the BAIR Commons HIC Center.

## Citation 
Please give us a star 🌟 on Github to support us!

Please cite our work if you find it inspiring or use our code in your work:
```
@article{niu2025pre,
  title={Pre-training auto-regressive robotic models with 4d representations},
  author={Niu, Dantong and Sharma, Yuvan and Xue, Haoru and Biamby, Giscard and Zhang, Junyi and Ji, Ziteng and Darrell, Trevor and Herzig, Roei},
  journal={arXiv preprint arXiv:2502.13142},
  year={2025}
}
```
