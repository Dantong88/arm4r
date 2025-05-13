from random import random
from typing import List
import os
import torch
import h5py
from yarr.agents.agent import Agent, ActResult, Summary
from scipy.spatial.transform import Rotation as R
import math
import numpy as np
from PIL import Image
from helpers import utils
import json
import matplotlib.pyplot as plt
from datetime import datetime
from agents.arm4r_bc.utils import *
from scipy.spatial.transform import Rotation
from collections import OrderedDict, deque
from arm4r.models.policy.arm4r_wrapper import ARM4RWrapper
import random
import open_clip

class Text_Tokenizer():
    def __init__(self):
        self.model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')

    def tokenize(self, text):
        # image = preprocess(Image.open("docs/CLIP.png")).unsqueeze(0)
        text = self.tokenizer([text])

        with torch.no_grad(), torch.cuda.amp.autocast():
            # image_features = model.encode_image(image)
            text_features = self.model.encode_text(text)

        return text_features

def get_data_from_h5(data, episode_name, resolution=(180, 320, 3), return_PIL_images=False):
    # img, action and proprio keys can be found in config/dataset_config_template.yaml
    side_images = data[f"{episode_name}/observation/exterior_image_1_left"][:]
    wrist_images = data[f"{episode_name}/observation/wrist_image_left"][:]

    action_keys = ["action/cartesian_position", "action/gripper_position"]
    proprio_keys = ["observation/cartesian_position", "observation/gripper_position"]
    actions = np.concatenate([data[f"{episode_name}/{key}"][:] for key in action_keys], axis=-1)
    proprios = np.concatenate([data[f"{episode_name}/{key}"][:] for key in proprio_keys], axis=-1)

    side_images_l, wrist_images_l = [], []
    selected_idx = []
    for idx, (si, wi) in enumerate(zip(side_images, wrist_images)):
        if len(si) == 0 or len(wi) == 0:
            continue
        side_images_l.append(np.frombuffer(si, dtype="uint8").reshape(resolution))
        wrist_images_l.append(np.frombuffer(wi, dtype="uint8").reshape(resolution))
        selected_idx.append(idx)

    # update the actions and proprios
    actions = actions[selected_idx]
    proprios = proprios[selected_idx]

    side_images_l = np.array(side_images_l)
    wrist_images_l = np.array(wrist_images_l)

    if return_PIL_images:
        side_images_l = [Image.fromarray(side_images_l[i]) for i in range(side_images_l.shape[0])]
        wrist_images_l = [Image.fromarray(wrist_images_l[i]) for i in range(wrist_images_l.shape[0])]

    return {
        "side_images": side_images_l,
        "wrist_images": wrist_images_l,
        "actions": actions,
        "proprios": proprios
    }

class ARM4RAgent(Agent):
    '''
    this is the model that predict 16 steps actions
    '''
    def __init__(self, debug = True):
        if debug != False:
            self.debug = True
        else:
            self.debug = False

        self.text_tokenizer = Text_Tokenizer()
        if self.debug:
            save_root = debug
            # Get the current date and time, formatted as yyyy-mm-dd-hh:mm:ss
            # folder_name = os.path.join(save_root, datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
            folder_name = os.path.join(save_root)
            os.makedirs(folder_name, exist_ok=True)
            self.save_dir = folder_name
            print(f"Folder '{folder_name}' created successfully!")
        self.gt = False

    def load_weights(self, savedir: str):
        checkpoint_path = savedir
        train_yaml_path = os.path.join(os.path.dirname(checkpoint_path), 'run.yaml')
        # print(os.getcwd())
        vision_encoder_path = "../../../../vision_encoder/cross-mae-rtx-vitb.pth"
        self.arm4r = ARM4RWrapper(train_yaml_path, checkpoint_path, vision_encoder_path)


    def reset(self, language_goal, eps_index):
        self.instruction = self.text_tokenizer.tokenize(language_goal)[0].numpy()
        # reset anf prompt the model
        self.control_error = {}
        self.control_error['success'] = False
        self.arm4r.reset()
        self.action = None

    def build(self, training: bool, device=None) -> None:
        self._device = device
        if self._device is None:
            self._device = torch.device('cpu')

    def update(self, step: int, replay_sample: dict) -> dict:
        priorities = 0
        total_losses = 0.
        for qa in self._qattention_agents:
            update_dict = qa.update(step, replay_sample)
            replay_sample.update(update_dict)
            total_losses += update_dict['total_loss']
        return {
            'total_losses': total_losses,
        }

    def act(self, eps_index: int, step: int, observation: dict,
            deterministic=False, delta = True) -> ActResult:

        image_front = np.transpose(observation['front_rgb'][-1], (1, 2, 0))
        # image_wrist = np.transpose(observation['overhead_rgb'][-1], (1, 2, 0))
        image_wrist = np.transpose(observation['wrist_rgb'][-1], (1, 2, 0))
        image_save = np.concatenate([image_front, image_wrist], axis=1)
        image_front = Image.fromarray(image_front)
        image_wrist = Image.fromarray(image_wrist)
        image_save = Image.fromarray(image_save)

        if self.debug:
            save_path = os.path.join(self.save_dir, str(eps_index), '{}.jpg'.format(step))
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.save_image(image_save, save_path, save_npy=False)
        if not self.gt:
            gripper_pos = observation['gripper_pos'][0]
            gripper_open = observation['gripper_open'][0]
            quat_rot = gripper_pos[-4:]

            euler = R.from_quat(quat_rot).as_euler("XYZ")
            gripper_position = np.array([gripper_open], dtype=np.float32).reshape(-1, 1)
            cartesian_position = np.concatenate([gripper_pos[:3].reshape(1, -1), euler.reshape(1, -1)], axis=1,
                                                dtype=np.float32)

            proprios = np.concatenate([cartesian_position, gripper_position], axis=-1)

            action = self.arm4r(
                image_front, image_wrist,
                proprios,
                instruction=self.instruction,
                action=self.action,
                use_temporal=True,
                teacher_forcing=False,
                binary_gripper = True,
            )

            self.action = proprios

            action_quat = self.euler_to_quat(action[3:6].reshape([1, -1]))[0]
            action_quat = np.concatenate([action[:3], action_quat, action[-1:]], axis=0)

            self.control_error[step] = {}
            self.control_error[step]['cmd'] = list(np.array(action, dtype=float))
            if step != 0:
                self.control_error[step - 1]['real_return'] = list(np.array(proprios[0], dtype=float))
                # diff = abs(proprios[0][:-1] - self.control_error[step]['cmd'][:-1])
                # print(diff)
        else:
            # load the gt
            try:
                gt_action = self.gt_actions[step]

                gripper_pos = observation['gripper_pos'][0]
                gripper_open = observation['gripper_open'][0]
                quat_rot = gripper_pos[-4:]

                euler = R.from_quat(quat_rot).as_euler("XYZ")
                gripper_position = np.array([gripper_open], dtype=np.float32).reshape(-1, 1)
                cartesian_position = np.concatenate([gripper_pos[:3].reshape(1, -1), euler.reshape(1, -1)], axis=1,
                                                    dtype=np.float32)

                proprios = np.concatenate([cartesian_position, gripper_position], axis=-1)
                self.control_error[step] = {}
                self.control_error[step]['cmd'] = list(np.array(gt_action, dtype=float))
                if step != 0:
                    self.control_error[step - 1]['real_return'] = list(np.array(proprios[0], dtype=float))
                    # diff = abs(proprios[0][:-1] - self.control_error[step]['cmd'][:-1])
                    # print(diff)

            except:
                gt_action = self.gt_actions[-1]
            action_quat = self.euler_to_quat(gt_action[3:6].reshape([1, -1]))[0]
            action_quat = np.concatenate([gt_action[:3], action_quat, gt_action[-1:]], axis=0)


        return ActResult(
            action_quat
        )

    def euler_to_quat(self, euler: np.ndarray, format_euler="XYZ", format_quat="xyzw"):
        """
        Convert euler angles to quaternion
        euler: N, 3
        """
        assert format_quat in ["wxyz", "xyzw"], "Invalid quaternion format, only support wxyz or xyzw"
        quat = Rotation.from_euler(format_euler, euler, degrees=False).as_quat()
        if format_quat == "wxyz":
            quat = quat[:, [3, 0, 1, 2]]
        return quat

    def convert_abs_action(self, action, proprio, use_quat = False):
        '''
        Calculate the next state from the delta action and the current proprioception
        action: S, T, action_dim
        proprio: S, T, proprio_dim
        '''
        delta_trans = action[ :, :3].reshape(-1, 3)
        delta_rot = action[:, 3:9].reshape(-1, 6)
        delta_rot = Rotation.from_matrix(rot_6d_to_rot_mat(delta_rot))

        current_state = np.repeat(proprio[0:1], action.shape[0], 0)
        current_trans = current_state[:, :3].reshape(-1, 3)
        current_rot = Rotation.from_matrix(rot_6d_to_rot_mat(current_state[ :, 3:9].reshape(-1, 6)))

        trans = np.einsum('ijk,ik->ij', current_rot.as_matrix(), delta_trans) + current_trans
        rot = (current_rot * delta_rot).as_matrix()

        rot = rot_mat_to_rot_6d(rot).reshape(action.shape[0], 6)
        trans = trans.reshape(action.shape[0], 3)

        if use_quat:
            rot = rot_6d_to_quat(rot)

        # process the gripper to be binary
        gripper_pred = action[:, -1:]
        gripper_pred = np.clip(np.round(gripper_pred), 0, 1)

        actions = np.concatenate([trans, rot, gripper_pred], axis=-1)


        return actions

    def save_image(self, img_pil, save_name, save_npy= False):

        # prev_front = np.load(save_front)

        # os.makedirs(os.path.dirname(save_name), exist_ok=True)
        img_pil.save(save_name)

        if save_npy:
            np.save(np.array(img_pil), save_name + '.npy')


    def update_summaries(self) -> List[Summary]:
        summaries = []
        for qa in self._qattention_agents:
            summaries.extend(qa.update_summaries())
        return summaries

    def act_summaries(self) -> List[Summary]:
        s = []
        return s


    def save_weights(self, savedir: str):
        for qa in self._qattention_agents:
            qa.save_weights(savedir)
