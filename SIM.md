## Simulation on RLBench

When using the 3d points pre-training model in a specific downstream task, i.e. robot manipulation tasks, we need to further 
perform robotic task tuning using collected demonstrations in the new environment setting. Here, we provide instructions for running this stage of the training on the [RLBench Benchmark](https://github.com/stepjam/RLBench).

Our code is built based mainly on [PerAct](https://github.com/peract/peract) and [RLBench](https://github.com/stepjam/RLBench), please consider citing them if you find their code useful!

The following content is structured into four parts:
* Simulation Environment Installation
* Prepare Simulation Data
* Robotics Task Tuning
* Inference

***
### Simulation Environment Installation
#### 1. You can still use the ``arm4r`` conda environment in pre-training.
```bash
conda activate arm4r
```
#### 2. PyRep and CoppeliaSim

Follow instructions from the official [PyRep](https://github.com/stepjam/PyRep) repo; reproduced here for convenience:

PyRep requires version **4.1** of CoppeliaSim. Download: 
- [Ubuntu 16.04](https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Player_V4_1_0_Ubuntu16_04.tar.xz)
- [Ubuntu 18.04](https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Player_V4_1_0_Ubuntu18_04.tar.xz)
- [Ubuntu 20.04](https://www.coppeliarobotics.com/previousVersions#)

Once you have downloaded CoppeliaSim, you can pull PyRep from git:

```bash
cd arm4r/sim
git clone https://github.com/stepjam/PyRep.git
cd PyRep
```

Add the following to your *~/.bashrc* file: (__NOTE__: the 'EDIT ME' in the first line)

```bash
export COPPELIASIM_ROOT=<EDIT ME>/PATH/TO/COPPELIASIM/INSTALL/DIR
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

Remember to source your bashrc (`source ~/.bashrc`) or 
zshrc (`source ~/.zshrc`) after this.

**Warning**: CoppeliaSim might cause conflicts with ROS workspaces. 

Finally install the python library:

```bash
pip install -r requirements.txt
pip install .
```

You should be good to go!
You could try running one of the examples in the *examples/* folder.

If you encounter errors, please use the [PyRep issue tracker](https://github.com/stepjam/PyRep/issues).

#### 3. RLBench

ARM4R uses my [RLBench fork](https://github.com/Dantong88/RLBench/tree/llarva_3dp). 

```bash
cd arm4r/sim
git clone -b llarva_3dp https://github.com/Dantong88/RLBench # note: 'llarva_3dp' branch

cd RLBench
pip install -r requirements.txt
python setup.py develop
```

For [running in headless mode](https://github.com/MohitShridhar/RLBench/tree/peract#running-headless), tasks setups, and other issues, please refer to the [official repo](https://github.com/stepjam/RLBench).

#### 4. YARR

AMR4R uses my [YARR fork](https://github.com/Dantong88/YARR/tree/llarva_3dp).

```bash
cd arm4r/sim
git clone -b llarva_3dp https://github.com/Dantong88/YARR # note: 'llarva_3dp' branch

cd YARR
pip install -r requirements.txt
python setup.py develop
```

Fianlly
```bash
cd arm4r/sim
pip install pip==24.0 # lower version of pip to support omegaconf==2.0.6
pip install -r requirements.txt
```

***

### Prepare Simulation Data

Then you need to generate the simulation demonstrations, which are used in our robotics
tuning.

#### 1. Generate RLBench Demos
Generate training demonstrations: (skip this if you only want to run eval)
```bash
cd arm4r/sim/RLBench/tools
export SIM_ROOT=YOUR_PATH_TO_arm4r/sim
python dataset_generator.py --tasks=meat_off_grill \
                            --save_path=$SIM_ROOT/data/train \
                            --image_size=128,128 \
                            --renderer=opengl \
                            --episodes_per_task=400 \
                            --processes=1 \
                            --all_variations=True
```

Note: you might need to create an off-screen display if running headless. You can do this with

```bash
sudo apt-get install xvfb
Xvfb :1 -screen 0 1024x768x24 &
export DISPLAY=:1
```

If you get a Qt platform plugin "xcb" error, [this](https://askubuntu.com/questions/1271976/could-not-load-the-qt-platform-plugin-xcb-in-even-though-it-was-found) is a useful thread for debugging the issue. 

Generate validation demonstrations: (skip this if you only want to run eval)
```bash
cd arm4r/sim/RLBench/tools
export SIM_ROOT=YOUR_PATH_TO_arm4r/sim
python dataset_generator.py --tasks=meat_off_grill \
                            --save_path=$SIM_ROOT/data/val \
                            --image_size=128,128 \
                            --renderer=opengl \
                            --episodes_per_task=25 \
                            --processes=1 \
                            --all_variations=True
```

After this, you will have generated 400 demos in ``$SIM_ROOT/data/train`` and 25 demos in ``$SIM_ROOT/data/val``. You may need to generate more demos for use in training.

#### 2. Convert RLBench Demos to Training Format
Assuming you have already generated 400 RLBench episodes for training, you can then convert the data to our supported format
by running:
```bash
cd arm4r/sim/process_datasets
python generate_training_eps.py # this will generate the 400eps.json
python convert_dataset_rlbench.py --eps_list your_generated_eps_json
```
Finally, you should get the data structure as:

```bash
arm4r/sim
│ 
└──process_datasets/annotations
│   ├── arm4r_format/meat_off_grill_400_front_wrist/rlbench_tasks/common_task/00000
│   │   ├── proprio.zarr 
│   │   │   └── (array of data with size T * 7 (x, y, z, yaw, roll, pitch, gripper))
│   │   ├── instruction.zarr 
│   │   │   └── (array of text embedding with size (768,))
│   │   └── action.zarr
│   │   │   └── (array of data with size T * 7 (x, y, z, yaw, roll, pitch, gripper))
│   │   └── images
│   │       └── images.json
│   │   
│   └── arm4r_format/meat_off_grill_400_front_wrist/rlbench_tasks/common_task/00001
│   │   
│   └── train/meat_off_grill_400
│   │   └── 400eps.json
│
└── data
│   ├── train
│   ├── val
└── ...
```


***
### Fine-tuning on the RLBench Demos
Your can easily launch the fine-tuning by first editing the ```dataset_dir``` field in ```arm4r/config/dataset_config_rlbench.json```, then running:
```bash
cd arm4r
export CUDA_VISIBLE_DEVICES=4,5,6,7
torchrun --nproc_per_node=4 --master_port=2453 scripts/finetune_robotics.py --dataset-cfg.dataset-json config/dataset_config_rlbench.json --logging-cfg.output-dir output --logging-cfg.log-name finetune_rlbench_meat_off_grill --optimizer-cfg.warmup-epochs 1.25 --trainer-cfg.epochs 125 --model-cfg.vision-encoder-cfg.vision-encoder arm4r-ckpts/vision_encoder/cross-mae-rtx-vitb.pth --dataset-cfg.num-repeat-traj 1 --model-cfg.policy-cfg.no-prompt-loss --model-cfg.policy-cfg.task robotics --model-cfg.policy-cfg.scratch-llama-config config/model_config/custom_transformer.json --dataset-cfg.non-overlapping 1 --shared-cfg.save-every 5 --dataset-cfg.shuffle-repeat-traj --optimizer-cfg.lr 5e-4 --shared_cfg.batch_size 64 --shared-cfg.seq_length 16 --model-cfg.policy-cfg.pretrained_path arm4r-ckpts/model_ckpts/pretrained_epic/pretrained_epic.pth
```



***
### Inference

To test the model on RLBench, just run

```bash
cd arm4r/sim
export SIM_ROOT=$(pwd)
python eval.py \
rlbench.tasks=[meat_off_grill] \
rlbench.demo_path=$SIM_ROOT/data/val \
framework.eval_from_eps_number=0 \
framework.eval_episodes=25 \
rlbench.episode_length=200 \
framework.gpu=4
```

Make sure you modify the [config file](arm4r/sim/conf/eval_arm4r.yaml)  for the fields ```method.policy_ckpt``` (either using our provide example ckpt or your trained ckpt with 400 demos),
and ```method.debug``` (a path to save all the step wise results for each episode). 

NOTE: for the checkpoint path, please use an absolute path.

🔸*For the checkpoints selection*: In the default training command, we set the total number of epochs
as 125. In most cases, training this long is not required, and performance can be judged based on validation. For reference, the example checkpoint we release for meat_off_grill is trained roughly for 50 epochs, and you should get around a 90% success rate for this task with your randomly generated validation set.


