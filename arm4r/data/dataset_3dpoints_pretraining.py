import os
import json
import h5py
import torch
import numpy as np
import torchvision.transforms as transforms
from typing import Union
from .utils import euler_to_rot_6d, quat_to_rot_6d, euler_to_quat, load_json, convert_multi_step, convert_delta_action_pretrain, \
    find_increasing_subsequences, create_prompt_mask, scale_action
from arm4r.util.args import DatasetConfig, SharedConfig
from collections import defaultdict
from glob import glob
import zarr
from PIL import Image


class SequenceDataset(torch.utils.data.Dataset):
    # set minimum trajectory length
    # we use 30 as the control frequency of the robot is 15 Hz
    minimum_length: int = 30
    maximum_length: int = 450

    # remove long tail situations
    min_demos: int = 4

    def __init__(
            self,
            dataset_config: DatasetConfig,
            shared_config: SharedConfig,
            vision_transform: transforms.Compose,
            no_aug_vision_transform: transforms.Compose = None,  # this is for wrist camera in particular
            split: str = "train",
            split_file: str = None,  # path to the train val split file (json)
    ):
        # parse the dataset config
        dataset_json = load_json(dataset_config.dataset_json)

        self.image_resolution = dataset_json['image_resolution']

        # shuffle repeat trajectory
        # if this is true, with half the chance the sequence of trajectories
        # tau_1, tau_1, tau_2 would turn into tau_1, tau_2, tau_1
        self.shuffle_repeat_traj = dataset_config.shuffle_repeat_traj
        if self.shuffle_repeat_traj:
            assert dataset_config.sort_by_lang, "Shuffle repeat trajectory only works with sort by lang"

        # calculate length of dataset
        self.dataset_root = dataset_json['dataset_dir']
        self.points_key = dataset_json['points_key']
        self.tasks = dataset_json['tasks']
        if shared_config.use_points_mask:
            self.use_points_mask = True
            self.points_mask_key = dataset_json['points_mask_key']
        else:
            self.use_points_mask = False

        common_path = glob(os.path.join(self.dataset_root, "**/*{}.zarr".format(self.points_key)), recursive=True)
        self.common_path = [os.path.join(self.dataset_root, p.replace("/{}.zarr".format(self.points_key), "")) for p in common_path]
        self.file2length = {}
        for file in self.common_path:
            self.file2length[file] = self.get_traj_length(file, self.points_key) - 1 # remove the last one

        self.hdf5_keys = self.common_path

        self.epi_len_mapping_json = self.file2length

        # filter episodes by their length
        self.hdf5_keys = [
            key for key in self.hdf5_keys if
            self.minimum_length <= self.epi_len_mapping_json[key] <= self.maximum_length
        ]

        self.verb_to_episode = defaultdict(list)

        # generate the verb_to_episode, if the task dist is empty, will load all the demos under the folder
        if len(self.tasks) >= 1:
            verbs = glob(os.path.join(self.dataset_root, ))
            for v in verbs:
                for _task in self.tasks:
                    eps = glob(os.path.join(v, '{}/*/*/*').format(_task))[:self.tasks[_task]]
                    self.verb_to_episode[os.path.basename(v)].extend(eps)
        else:
            verbs = glob(os.path.join(self.dataset_root, '*/*'))
            # verbs = [os.path.basename(v) for v in verbs]
            for v in verbs:
                eps = glob(os.path.join(v, '*'))
                self.verb_to_episode[os.path.basename(v)].extend(eps)

        # confine training to only a subset of tasks
        if dataset_config.task_names:
            # first check if the task names are valid
            for task in dataset_config.task_names:
                assert task in self.verb_to_episode, f"Task {task} not found in the dataset"
            self.verb_to_episode = {k: v for k, v in self.verb_to_episode.items() if k in dataset_config.task_names}
            # remove episodes that are not in the hdf5 keys use set intersection
            self.hdf5_keys = set(self.hdf5_keys).intersection(
                set([item for sublist in self.verb_to_episode.values() for item in sublist])
            )
            self.hdf5_keys = list(self.hdf5_keys)

        # sort the hdf5 keys so that the permutation is consistent
        self.hdf5_keys = sorted(self.hdf5_keys)

        if dataset_config.dataset_fraction < 1.0:
            print("Using only a fraction of the dataset: ", dataset_config.dataset_fraction)
            if not dataset_config.sort_by_lang:
                # if sort_by_lang, process the dataset_fraction by task
                num_demos = int(len(self.hdf5_keys) * dataset_config.dataset_fraction)
                self.hdf5_keys = self.hdf5_keys[:num_demos]

        # define train test split
        self.split = split
        self.train_split = dataset_json["train_split"]

        # set seed and shuffle the hdf5 keys
        rng = np.random.RandomState(seed=shared_config.seed)
        rng.shuffle(self.hdf5_keys)
        num_train = int(len(self.hdf5_keys) * self.train_split)

        if split_file is not None:
            with open(split_file, 'r') as f:
                self.hdf5_keys = json.load(f)
        else:
            if self.split == "train":
                self.hdf5_keys = self.hdf5_keys[:num_train]
            else:
                self.hdf5_keys = self.hdf5_keys[num_train:]

        # if sort by lang, we first shuffle the task permutation and then the episodes
        # this ensures that for most indices, there's no overlap between tasks
        self.sort_by_lang = dataset_config.sort_by_lang

        if self.split == "val":
            self.min_demos = 1

        if self.sort_by_lang:
            # remove episodes that are not in the hdf5 keys use set intersection
            all_keys = set(self.hdf5_keys)
            self.verb_to_episode = {k: list(set(v).intersection(all_keys)) for k, v in self.verb_to_episode.items()}
            # remove all verbs that have length shorter than min_demos
            self.verb_to_episode = {k: sorted(v) for k, v in self.verb_to_episode.items() if len(v) >= self.min_demos}

            # ablation: use only a fraction of the dataset
            if dataset_config.dataset_fraction < 1.0:
                for k in self.verb_to_episode:
                    num_demos = max(int(len(self.verb_to_episode[k]) * dataset_config.dataset_fraction), self.min_demos)
                    self.verb_to_episode[k] = self.verb_to_episode[k][:num_demos]
                self.hdf5_keys = [item for sublist in self.verb_to_episode.values() for item in sublist]

            # to support task_barrier, we need to calculate how many steps are available for each verb/task
            # this is constant for each verb/task
            self.verb_to_numsteps = {
                k: sum([self.epi_len_mapping_json[epi] for epi in v]) for k, v in self.verb_to_episode.items()
            }

        # rebalance tasks
        self.rebalance_tasks = dataset_config.rebalance_tasks
        if self.rebalance_tasks:
            assert self.sort_by_lang, "Rebalance tasks only works with sort by lang"
            # calculate median of the number of trajectories for each task
            if self.split == "train":
                self.rebalance_length = int(np.median([len(i) for i in self.verb_to_episode.values()]))
                # self.rebalance_length = int(np.quantile([len(i) for i in self.verb_to_episode.values()], 0.75)) # for droid pre-training
            else:
                # self.rebalance_length = 5
                self.rebalance_length = int(np.median([len(i) for i in self.verb_to_episode.values()]))
            print("Each task is rebalanced to have length: ", self.rebalance_length)

        # define image, proprio, and action keys
        self.image_keys = dataset_json["image_keys"]
        self.proprio_keys = dataset_json["proprio_keys"]
        self.action_keys = dataset_json["action_keys"]
        self.instruction_key = dataset_json["instruction_key"]
        self.num_points = dataset_json["num_points"]

        # define sequence length
        self.seq_length = shared_config.seq_length

        self.num_weighted_steps = dataset_config.num_weighted_steps

        self.goal_conditioned = dataset_config.goal_conditioned

        # define rotation format
        self.rot_6d = shared_config.rot_6d

        # non overlapping subsequence?
        self.non_overlapping: Union[bool, int] = dataset_config.non_overlapping

        # enable repeating trajectory so that it can learn the copying behavior
        self.num_repeat_traj = dataset_config.num_repeat_traj

        # define use delta action flag
        self.use_delta_action = shared_config.use_delta_action

        self.task_barrier = dataset_config.task_barrier
        self.skip_step = dataset_config.skip_step
        self.proprio_noise = dataset_config.proprio_noise
        self.action_noise = dataset_config.action_noise

        # vision transform
        # we do not need normalization, see get_item
        if vision_transform is not None:
            self.vision_transform = transforms.Compose([t for t in vision_transform.transforms if
                                                        not isinstance(t, transforms.ToTensor) and not isinstance(t,
                                                                                                                  transforms.ColorJitter)])
        else:
            print("warning: vision transforms are not defined. Using default transforms.")
            self.vision_transform = transforms.Compose([
                transforms.Resize(size=248, max_size=None, interpolation=transforms.InterpolationMode.BICUBIC,
                                  antialias='warn'),  # kept consistent with default
                transforms.CenterCrop(size=224),
                transforms.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]),
                                     std=torch.tensor([0.2290, 0.2240, 0.2250]))
            ])
        if no_aug_vision_transform is not None:
            self.no_aug_vision_transform = transforms.Compose([t for t in no_aug_vision_transform.transforms if
                                                               not isinstance(t,
                                                                              transforms.ToTensor) and not isinstance(t,
                                                                                                                      transforms.ColorJitter)])
        else:
            self.no_aug_vision_transform = self.vision_transform
        print("vision transforms")
        print(self.vision_transform)

        if dataset_config.vision_aug:
            self.vision_aug = True
            self.contrast_range = [0.8, 1.2]
            self.brightness_range = [-0.1, 0.1]
            print("using numeric brightness and contrast augmentation")
            print("contrast range: ", self.contrast_range)
            print("brightness range: ", self.brightness_range)
        else:
            self.vision_aug = False

        # change prediction to be k steps
        self.num_pred_steps = shared_config.num_pred_steps
        assert self.num_pred_steps >= 1, "Number of prediction steps must be at least 1"
        print("Number of prediction steps: ", self.num_pred_steps)

        # rebalance the dataset with respect to the number of tasks in each group. The grouping is calculated so that
        # each group is repeated the same number of times
        self.task_grouping = dataset_json.get("task_grouping", None)
        if self.task_grouping is not None:
            task_grouping = json.load(open(self.task_grouping, 'r'))
            average_num_tasks = np.mean([len(v) for v in task_grouping["tasks"].values()])
            print("Average number of tasks: ", average_num_tasks)
            self.upweight_tasks = {}

            ratios = task_grouping.get("ratios", None)
            if ratios is not None:
                print("overriding with known ratio: ", ratios)
                for k, task_lists in task_grouping["tasks"].items():
                    task_ratio = ratios[k]
                    for t in task_lists:
                        self.upweight_tasks[t] = task_ratio
            else:
                for _, task_lists in task_grouping["tasks"].items():
                    task_len = len(task_lists)
                    upweight_factor = average_num_tasks / task_len
                    for t in task_lists:
                        self.upweight_tasks[t] = upweight_factor

        # load the dataset
        self.shuffle_dataset(seed=0)

    def get_traj_length(
            self,
            file_path: str,
            key: str,
    ):
        action_fp = file_path + "/{}.zarr".format(key)
        actions = zarr.load(action_fp)
        return len(actions)

    def total_seq_length(self):
        total_seq_length = 0
        for key in self.hdf5_keys:
            total_seq_length += self.epi_len_mapping_json[key]
        return total_seq_length

    def update_seq_length(self, new_seq_length: int):
        """
        Update the sequence length
        """
        self.seq_length = new_seq_length

    def save_split(self, path: str):
        """
        Save the train test split to a json file
        """
        with open(path, 'w') as f:
            json.dump(self.hdf5_keys, f)

    def shuffle_dataset(self, seed=0):
        if self.goal_conditioned:
            self.shuffle_dataset_goal_conditioned(seed)
        elif self.sort_by_lang:
            self.shuffle_dataset_sort_by_lang(seed)
        else:
            self.shuffle_dataset_default(seed)

    def shuffle_dataset_default(self, seed=0):
        """
        Shuffle the dataset according to the seed
        """
        rng = np.random.RandomState(seed=seed)
        # shuffle the permutation of the episodes
        indices = rng.permutation(len(self.hdf5_keys))
        self.steps = []
        for i in indices:
            key = self.hdf5_keys[i]
            num_repeat = rng.choice(np.arange(self.num_repeat_traj)) + 1
            for _ in range(num_repeat):
                for s in range(self.epi_len_mapping_json[key]):
                    self.steps.append(
                        {
                            "episode_id": key,
                            "step": s,
                            "eos": s == self.epi_len_mapping_json[key] - 1,
                        }
                    )

    def shuffle_dataset_goal_conditioned(self, seed=0):
        """
        Shuffle the dataset according to the seed
        """
        rng = np.random.RandomState(seed=seed)
        # first we shuffle the verbs
        verbs = list(self.verb_to_episode.keys())
        rng.shuffle(verbs)

        self.steps = []
        all_keys = []
        for v in verbs:
            # shuffle the episode ids
            if self.rebalance_tasks:
                # update rebalance length based on task grouping if defined
                rl = self.rebalance_length
                if self.task_grouping is not None:
                    rl = int(rl * self.upweight_tasks[v])
                if len(self.verb_to_episode[v]) < rl:
                    replace = True
                else:
                    replace = False
                indices = rng.choice(len(self.verb_to_episode[v]), size=rl, replace=replace)
            else:
                indices = rng.permutation(len(self.verb_to_episode[v]))
            episode_keys = [self.verb_to_episode[v][i] for i in indices]
            all_keys.extend(episode_keys)
        rng.shuffle(all_keys)
        self.steps = []
        self.usable_indices = []
        for key in all_keys:
            for s in range(self.epi_len_mapping_json[key]):
                self.steps.append(
                    {
                        "episode_id": key,
                        "step": s,
                        "eos": s == self.epi_len_mapping_json[key] - 1,
                    }
                )
                if s == 0:
                    self.usable_indices.append(len(self.steps) - 1)

    def shuffle_dataset_sort_by_lang(self, seed=0):
        """
        Shuffle the dataset according to the seed
        """
        rng = np.random.RandomState(seed=seed)
        # first we shuffle the verbs
        verbs = list(self.verb_to_episode.keys())
        rng.shuffle(verbs)

        self.steps = []
        verb_to_idx = defaultdict(list)
        for v in verbs:
            # shuffle the episode ids
            if self.rebalance_tasks:
                # update rebalance length based on task grouping if defined
                rl = self.rebalance_length
                if self.task_grouping is not None:
                    rl = int(rl * self.upweight_tasks[v])
                if len(self.verb_to_episode[v]) < rl:
                    replace = True
                else:
                    replace = False
                indices = rng.choice(len(self.verb_to_episode[v]), size=rl, replace=replace)
            else:
                indices = rng.permutation(len(self.verb_to_episode[v]))

            repeats = rng.choice(np.arange(self.num_repeat_traj), size=indices.shape[0], replace=True) + 1
            episode_keys = [self.verb_to_episode[v][i] for i in indices]
            task_length = sum([self.epi_len_mapping_json[key] * r for key, r in zip(episode_keys, repeats)])
            if task_length < self.seq_length:
                continue

            # initialize the current step index for the verb
            verb_step_idx = 0
            cache = []
            ranges = []

            # calculate the ranges of trajectories in index space
            start_idx = 0
            for key, num_repeat in zip(episode_keys, repeats):
                trajectory_length = self.epi_len_mapping_json[key]
                for repeat_i in range(num_repeat):
                    ranges.append((start_idx, start_idx + trajectory_length))  # (inclusive, exclusive)
                    start_idx += trajectory_length

            for key, num_repeat in zip(episode_keys, repeats):
                for repeat_i in range(num_repeat):
                    for s in range(self.epi_len_mapping_json[key]):
                        cache.append(
                            {
                                "episode_id": key,
                                "step": s,
                                "eos": s == self.epi_len_mapping_json[key] - 1,
                            }
                        )
                        # if task_barrier = True, it ensures that within each batch, there is only one verb/task
                        # if verb_step_idx + self.seq_length == self.verb_to_numsteps, it means that
                        # we have reached the end of the task, and we shouldn't include the next step
                        if self.task_barrier and verb_step_idx + self.seq_length > task_length:
                            continue
                        else:
                            # update the verb to idx mapping
                            verb_to_idx[v].append(len(self.steps) + verb_step_idx)
                        verb_step_idx += 1

            # shuffle cache based on ranges
            # we first shuffle range as randomly switch the consecutive two trajectories
            if self.shuffle_repeat_traj:
                for i in range(len(ranges) - 1):
                    if rng.uniform() < 0.5:
                        ranges[i], ranges[i + 1] = ranges[i + 1], ranges[i]
                # then shuffle the cache based on the ranges
                cache = [cache[i] for r in ranges for i in range(r[0], r[1])]
            self.steps.extend(cache)
        self.usable_indices = [idx for v in verb_to_idx.values() for idx in v]

    def __len__(self):
        """
        Return the length of the dataset
        """
        if (self.sort_by_lang and self.task_barrier) or self.goal_conditioned:
            # then only use the indices that are usable
            data_length = len(self.usable_indices)
        else:
            data_length = len(self.steps) - self.seq_length + 1

        if self.non_overlapping:
            if isinstance(self.non_overlapping, bool):
                new_data_length = data_length // self.seq_length
            else:
                new_data_length = data_length // self.non_overlapping

            if data_length < self.seq_length and data_length > 0:
                data_length = 1
            else:
                data_length = new_data_length

        return data_length

    def __getitem__(self, index):
        """
        Get the subsequence of the dataset starting from index to index + sequence_length
        return a diction of shape
        {
            "observation": torch.Tensor, shape (seq_length, num_cameras, 3, 224, 224)
            "proprio": torch.Tensor, shape (seq_length, num_pred_steps, proprio_dim)
            "action": torch.Tensor, shape (seq_length, num_pred_steps, action_dim)
        }
        """
        if self.non_overlapping:
            if isinstance(self.non_overlapping, bool):
                index = index * self.seq_length
            else:
                index = index * self.non_overlapping

        if (self.sort_by_lang and self.task_barrier) or self.goal_conditioned:
            # use self.usable_indices to map index to a subsequence that only contains one task
            index = self.usable_indices[index]

        subseq = self.steps[index: index + self.seq_length + self.num_pred_steps - 1]
        eos = np.array([s["eos"] for s in subseq])  # (seq_length + num_pred_steps - 1, 1)
        eos = torch.from_numpy(eos).float()
        start_end_epi = defaultdict(list)
        obs_start_end_epi = defaultdict(list)
        for idx, s in enumerate(subseq):
            start_end_epi[s["episode_id"]].append(s["step"])
            if idx < self.seq_length:
                obs_start_end_epi[s["episode_id"]].append(s["step"])

        start_end_epi = {
            k: find_increasing_subsequences(v) for k, v in start_end_epi.items()
        }
        obs_start_end_epi = {
            k: find_increasing_subsequences(v) for k, v in obs_start_end_epi.items()
        }

        proprio, action = self.helper_load_points(start_end_epi, k=self.points_key, num_points=self.num_points)
        instruction = self.helper_load_instruction(start_end_epi)  # (seq_length + num_pred_steps - 1, proprio_dim)
        # concatenate eos to action
        action = torch.cat([action, eos[:, None]], dim=-1)  # (seq_length, action_dim)
        if self.use_points_mask:
            points_mask = self.helper_load_points_mask(start_end_epi)
        else:
            points_mask = torch.ones([action.shape[0], proprio.shape[1]//3])

        # process action and proprio so that they are multi step prediction
        proprio = self.convert_multi_step(proprio, eos)[:self.seq_length]  # (seq_length, num_pred_steps, proprio_dim)
        action = self.convert_multi_step(action, eos)[:self.seq_length]  # (seq_length, num_pred_steps, action_dim)
        instruction = self.convert_multi_step_instruction(instruction, eos)[:self.seq_length]
        points_mask = self.convert_multi_step_points_mask(points_mask, eos)[:self.seq_length]

        if self.use_delta_action:
            if not self.rot_6d:
                print("Warning: use_delta_action is set to True, but rot_6d is set to False. This is not supported.")
            else:
                action = convert_delta_action_pretrain(action.numpy(), proprio.numpy())
                action = torch.from_numpy(action).float()

        observation = self.helper_load_image(obs_start_end_epi)

        # if we are goal conditioned, we need to add the goal to the observation
        if self.goal_conditioned:
            # find the first eos position
            eos_idx = np.where(eos.numpy() == 1)[0][0]

            # #find the first non zero observation
            # valid_obs = torch.any(observation.reshape(observation.shape[0],-1),1)
            # last_obs_idx = sorted(torch.where(valid_obs)[0])[-1]

            # eos_idx = min(eos_idx, last_obs_idx-1)
            # eos_idx = max(eos_idx, 0)

            thres_obs = torch.ones_like(observation[0]) * observation[eos_idx, 0, 0, 0, 0]
            while torch.allclose(observation[eos_idx, 0, 0], thres_obs, atol=1e-3):
                eos_idx -= 1
                thres_obs = torch.ones_like(observation[0]) * observation[eos_idx, 0, 0, 0, 0]
                if eos_idx == 0:
                    break

            if observation.shape[0] < self.seq_length:
                pad_len = self.seq_length - observation.shape[0]
                observation_pad = observation[-1:].repeat(pad_len, 1, 1, 1, 1)
                proprio_pad = proprio[-1:].repeat(pad_len, 1, 1)
                action_pad = action[-1:].repeat(pad_len, 1, 1)

                observation = torch.cat([observation, observation_pad], dim=0)
                proprio = torch.cat([proprio, proprio_pad], dim=0)
                action = torch.cat([action, action_pad], dim=0)

            observation[eos_idx:] = observation[eos_idx]
            proprio[eos_idx:] = proprio[eos_idx]
            action[eos_idx:] = action[eos_idx]

            observation = torch.cat([observation[eos_idx:eos_idx + 1], observation], dim=0)[:self.seq_length]
            proprio = torch.cat([proprio[eos_idx:eos_idx + 1], proprio], dim=0)[:self.seq_length]
            action = torch.cat([action[eos_idx:eos_idx + 1], action], dim=0)[:self.seq_length]

            prompt_mask = torch.zeros(self.seq_length)
            prompt_mask[1:eos_idx + 1] = 1
            weight_mask = torch.zeros(self.seq_length)
            if self.num_weighted_steps >= 1:
                self.num_weighted_steps = int(self.num_weighted_steps)
                weighted_step = min(self.num_weighted_steps, eos_idx)
                weight_mask[1:weighted_step + 1] = 1
            else:
                # num_step is a ratio
                seq_len = eos_idx
                weight_steps = int(self.num_weighted_steps * seq_len)
                weight_mask[1:weight_steps + 1] = 1

        else:
            prompt_mask, weight_mask = create_prompt_mask(action[..., 0, -1], self.num_weighted_steps)
            prompt_mask = torch.from_numpy(prompt_mask).float()
            weight_mask = torch.from_numpy(weight_mask).float()
        return {
            "observation": observation, # length * 2 * 3 * 224 * 224
            "proprio": proprio, # length * pred_step * 3888
            "action": action, # length * pred_step * 3889
            "instruction": instruction, # length * 512
            "prompt_mask": prompt_mask, # length
            "weight_mask": weight_mask, # length
            "points_mask": points_mask # length * 1296
        }

    def convert_multi_step(self, data: torch.Tensor, eos: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Convert the data for multi step prediction

        Args:
            action (torch.Tensor): the action to convert, of shape (seq_length, dim)

        Returns:
            torch.Tensor: the converted action, of shape (seq_length, num_pred_steps, dim)
        """
        if self.num_pred_steps == 1:
            return data.unsqueeze(1)
        if isinstance(eos, torch.Tensor):
            eos = eos.numpy()
        pos = np.concatenate(
            [np.array([0]), np.nonzero(eos)[0] + 1, np.array([self.seq_length + self.num_pred_steps - 1])])
        data_chunked = []
        for i in range(1, len(pos)):
            demo_start = pos[i - 1]
            demo_end = pos[i]
            data_chunked.append(convert_multi_step(data[demo_start: demo_end], self.num_pred_steps))
        return torch.cat(data_chunked, dim=0)

    def convert_multi_step_instruction(self, data: torch.Tensor, eos: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Convert the data for multi step prediction

        Args:
            action (torch.Tensor): the action to convert, of shape (seq_length, dim)

        Returns:
            torch.Tensor: the converted action, of shape (seq_length, num_pred_steps, dim)
        """
        if self.num_pred_steps == 1:
            return data #.unsqueeze(1)
        if isinstance(eos, torch.Tensor):
            eos = eos.numpy()
        pos = np.concatenate(
            [np.array([0]), np.nonzero(eos)[0] + 1, np.array([self.seq_length + self.num_pred_steps - 1])])
        data_chunked = []
        for i in range(1, len(pos)):
            demo_start = pos[i - 1]
            demo_end = pos[i]
            data_chunked.append(data[demo_start: demo_end])
        return torch.cat(data_chunked, dim=0)

    def convert_multi_step_points_mask(self, data: torch.Tensor, eos: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Convert the data for multi step prediction

        Args:
            action (torch.Tensor): the action to convert, of shape (seq_length, dim)

        Returns:
            torch.Tensor: the converted action, of shape (seq_length, num_pred_steps, dim)
        """
        if self.num_pred_steps == 1:
            return data #.unsqueeze(1)
        if isinstance(eos, torch.Tensor):
            eos = eos.numpy()
        pos = np.concatenate(
            [np.array([0]), np.nonzero(eos)[0] + 1, np.array([self.seq_length + self.num_pred_steps - 1])])
        data_chunked = []
        for i in range(1, len(pos)):
            demo_start = pos[i - 1]
            demo_end = pos[i]
            data_chunked.append(data[demo_start: demo_end])
        return torch.cat(data_chunked, dim=0)

    def helper_load_proprio(self, start_end_epi):
        """
        Load proprioception data from the dataset
        """

        # load proprioception data
        proprio = {}
        for k in self.proprio_keys:
            data = []
            for epi in start_end_epi:
                for s, e in start_end_epi[epi]:
                    data.append(self.get_key_from_demo_v2(epi, k, s, e))
            proprio[k] = np.concatenate(data, axis=0)

        ret = proprio[self.proprio_keys[0]]

        if self.proprio_noise > 0:
            # adding noise to proprio
            ret += np.random.normal(0, self.proprio_noise, ret.shape)
            if ret.shape[1] == 7:
                # normalize the quaternion
                ret[:, 3:] /= np.linalg.norm(ret[:, 3:], axis=-1, keepdims=True)

        rot = ret[:, 3:]
        # deal with rot_6d
        if self.rot_6d:
            if rot.shape[1] == 4:
                # robomimic dataset has format wxyz
                rot = quat_to_rot_6d(rot)
            elif rot.shape[1] == 3:
                rot = euler_to_rot_6d(rot)
            ret = np.concatenate([ret[:, :3], rot], axis=-1)
            proprio[self.proprio_keys[0]] = ret
        else:
            if rot.shape[1] == 3:
                # convert to quaternion (only happens for droid, which uses XYZ as the rotation format)
                rot = euler_to_quat(rot)
                # update the proprio
                ret = np.concatenate([ret[:, :3], rot], axis=-1)
                proprio[self.proprio_keys[0]] = ret
        proprio_vec = np.concatenate([proprio[k] for k in self.proprio_keys], axis=-1)
        proprio_vec = torch.from_numpy(proprio_vec).float()
        return proprio_vec

    def helper_load_instruction(self, start_end_epi):
        """
        Load proprioception data from the dataset
        """

        # load proprioception data
        proprio = {}
        for k in self.instruction_key:
            data = []
            for epi in start_end_epi:
                for s, e in start_end_epi[epi]:
                    data.append(self.get_key_from_demo_v2(epi, k, s, e))
            proprio[k] = np.concatenate(data, axis=0)

        return torch.from_numpy(proprio['instruction']).float()

    def helper_load_points_mask(self, start_end_epi):
        """
        Load proprioception data from the dataset
        """

        # load proprioception data
        points_mask = {}
        data = []
        for epi in start_end_epi:
            for s, e in start_end_epi[epi]:
                data.append(self.get_key_from_demo_v2(epi, self.points_mask_key, s, e))
            points_mask['points_mask'] = np.concatenate(data, axis=0)

        return torch.from_numpy(points_mask['points_mask']).float()

    def helper_load_action(self, start_end_epi):
        """
        Load action data from the dataset
        """
        action = {}
        for k in self.action_keys:
            data = []
            for epi in start_end_epi:
                for s, e in start_end_epi[epi]:
                    data.append(self.get_key_from_demo_v2(epi, k, s, e))
            action[k] = np.concatenate(data, axis=0)

        if self.action_noise > 0:
            # add noise to the action
            ret = action[self.action_keys[0]]
            ret += np.random.normal(0, self.action_noise, ret.shape)
            action[self.action_keys[0]] = ret

        if self.rot_6d:
            ret = action[self.action_keys[0]]
            rot = ret[:, 3:]
            rot = euler_to_rot_6d(rot)
            ret = np.concatenate([ret[:, :3], rot], axis=-1)
            action[self.action_keys[0]] = ret

        action_vec = np.concatenate([action[k] for k in self.action_keys], axis=-1)
        action_vec = torch.from_numpy(action_vec).float()
        return action_vec

    def helper_load_points(self, start_end_epi, k='front', num_points = 36):
        """
        Load proprioception data from the dataset
        """

        # load proprioception data
        data_proprio = []
        data_action = []
        for epi in start_end_epi:
            for s, e in start_end_epi[epi]:
                data_proprio.append(self.load_points(epi, k, s, e, num_points))
                data_action.append(self.load_points(epi, k, s + 1, e + 1, num_points))
        points_proprio = np.concatenate(data_proprio, axis=0)
        points_action = np.concatenate(data_action, axis=0)

        points_proprio_vec = torch.from_numpy(points_proprio).float()
        points_proprio_vec = points_proprio_vec.reshape(points_proprio_vec.shape[0], -1)

        points_action_vec = torch.from_numpy(points_action).float()
        points_action_vec = points_action_vec.reshape(points_action_vec.shape[0], -1)

        return points_proprio_vec, points_action_vec

    def helper_load_image(self, start_end_epi):
        """
        Load image data from the dataset
        """
        image = {}
        dtype = None
        for k in self.image_keys:
            data = []
            for epi in start_end_epi:
                for s, e in start_end_epi[epi]:
                    subsequence = self.get_key_from_demo_v2(epi, k, s, e)
                    if dtype is None:
                        dtype = subsequence.dtype
                        if dtype == 'uint8':
                            norm = 255.0
                        else:
                            norm = 1.0
                    subsequence = torch.from_numpy(subsequence / norm)
                    # data aug for brightness and contrast
                    if self.vision_aug:
                        contrast = np.random.uniform(self.contrast_range[0], self.contrast_range[1])
                        brightness = np.random.uniform(self.brightness_range[0], self.brightness_range[1])
                        subsequence = contrast * subsequence + brightness

                    # permute from T, H, W, C to T, C, H, W
                    subsequence = subsequence.permute(0, 3, 1, 2)
                    # transform each subsequence independently
                    if "wrist" in k or "hand" in k:
                        subsequence = self.no_aug_vision_transform(subsequence).float()
                    else:
                        subsequence = self.vision_transform(subsequence).float()
                    data.append(subsequence)
            image[k] = torch.cat(data, dim=0)  # concat on the time axis

        image_vec = torch.stack([image[k] for k in self.image_keys], dim=1).float()
        return image_vec

    def load_points(self, epi, k, s, e, num_points):
        file = os.path.join(epi, '{}.zarr'.format(self.points_key))
        points = zarr.load(file)[s:e + 1, :num_points]
        standard_points = np.zeros([points.shape[0], num_points, 3])
        standard_points[:, :points.shape[1], :] = points
        standard_points[:, :, 0] *= 0.002
        standard_points[:, :, 1] *= 0.002
        return standard_points

    def get_key_from_demo(
            self,
            demo_id: str,
            key: str,
            seq_begin_index: int,
            seq_end_index: int,
    ) -> np.ndarray:
        """
        Get the key from the demo
        Args:
            demo_id: str, the id of the demo
            key: str, the key to get from the demo
            seq_begin_index: int, the beginning index of the sequence
            seq_end_index: int, the ending index of the sequence (inclusive)

        Returns:
            np.ndarray, the data from the demo
        """
        # obtain the hdf5 file handle
        f_handle = self.keys_to_file[demo_id]
        # get the data from the hdf5 file
        if key == 'instruction':
            data = [f_handle[demo_id][key][()] for i in range(seq_begin_index, seq_end_index + 1)]
        else:
            data = f_handle[demo_id][key][seq_begin_index:seq_end_index + 1]
        if 'image' in key:
            if 'episode' in demo_id:
                data = np.frombuffer(data, dtype='uint8').reshape(-1, 128, 128, 3)
                # data = np.frombuffer(data, dtype='uint8').reshape(-1,180,320,3)
            elif 'demo' in demo_id:
                data = np.frombuffer(data, dtype='uint8').reshape(-1, 84, 84, 3)
            else:
                raise ValueError("Unknown demo type")
        return data

    def get_key_from_demo_v2(
            self,
            demo_id: str,
            key: str,
            seq_begin_index: int,
            seq_end_index: int,
    ) -> np.ndarray:
        """
        Get the key from the demo
        Args:
            demo_id: str, the id of the demo
            key: str, the key to get from the demo
            seq_begin_index: int, the beginning index of the sequence
            seq_end_index: int, the ending index of the sequence (inclusive)

        Returns:
            np.ndarray, the data from the demo
        """
        # obtain the hdf5 file handle
        if key == 'observation/cartesian_position':
            proprio_fp = os.path.join(demo_id, "proprio.zarr")
            # get proprio data
            data = zarr.load(proprio_fp)[seq_begin_index: seq_end_index + 1][:, :-1]
        elif key == 'observation/gripper_position':
            proprio_fp = os.path.join(demo_id, "proprio.zarr")
            # get proprio data
            data = zarr.load(proprio_fp)[seq_begin_index: seq_end_index + 1][:, -1:]
        elif key == 'action/cartesian_position':
            action_fp = os.path.join(demo_id, "action.zarr")
            # get proprio data
            data = zarr.load(action_fp)[seq_begin_index: seq_end_index + 1][:, :-1]
        elif key == 'action/gripper_position':
            action_fp = os.path.join(demo_id, "action.zarr")
            # get proprio data
            data = zarr.load(action_fp)[seq_begin_index: seq_end_index + 1][:, -1:]
        elif key == 'observation/exterior_image_1_left':
            camera_observations = []
            for img_dix in range(seq_begin_index, seq_end_index + 1, 1):
                try:
                    if os.path.exists(os.path.join(demo_id, 'images')):
                        current_frame_path = os.path.join(demo_id, 'images/exterior_image_1_left', '%05d.jpg' % img_dix)
                        current_frame = Image.open(current_frame_path)
                    else:
                        image_json = json.load(open(os.path.join(demo_id, 'images.json')))
                        current_frame_path = image_json[key][img_dix]
                        current_frame = Image.open(current_frame_path)
                except:
                    current_frame = np.random.randint(0, 255, self.image_resolution[0][1:], dtype="uint8")
                camera_observations.append(np.asarray(current_frame))
            camera_observations = np.stack(camera_observations)
            # data = np.reshape(camera_observations,[-1, 180, 320, 3])
            data = np.reshape(camera_observations, self.image_resolution[0])
        elif key == 'observation/ego_image':
            camera_observations = []
            for img_dix in range(seq_begin_index, seq_end_index + 1, 1):
                try:
                    if os.path.exists(os.path.join(demo_id, 'images')):
                        current_frame_path = os.path.join(demo_id, 'images/ego_image', '%05d.jpg' % img_dix)
                    else:
                        image_json = json.load(open(os.path.join(demo_id, 'images.json')))
                        current_frame_path = image_json[key][img_dix]

                    if not os.path.exists(current_frame_path):
                        raise FileNotFoundError

                    target_h, target_w, _ = self.image_resolution[0][1:]
                    current_frame = Image.open(current_frame_path).convert("RGB").resize((target_w, target_h))
                    current_frame_np = np.asarray(current_frame)

                    if current_frame_np.shape != (target_h, target_w, 3):
                        raise ValueError("Image has incorrect shape")
                except:
                    current_frame_np = np.random.randint(0, 255, self.image_resolution[0][1:], dtype="uint8")
                camera_observations.append(current_frame_np)
            camera_observations = np.stack(camera_observations)
            data = np.reshape(camera_observations, self.image_resolution[0])
        elif key == 'observation/wrist_image_left':
            camera_observations = []
            for img_dix in range(seq_begin_index, seq_end_index + 1, 1):
                try:
                    if os.path.exists(os.path.join(demo_id, 'images')):
                        current_frame_path = os.path.join(demo_id, 'images/wrist_image_left', '%05d.jpg' % img_dix)
                        current_frame = Image.open(current_frame_path)
                    else:
                        image_json = json.load(open(os.path.join(demo_id, 'images.json')))
                        current_frame_path = image_json[key][img_dix]
                        current_frame = Image.open(current_frame_path)
                except:
                    current_frame = np.random.randint(0, 255, self.image_resolution[1][1:], dtype="uint8")
                camera_observations.append(np.asarray(current_frame))
            camera_observations = np.stack(camera_observations)
            # data = np.reshape(camera_observations, [-1, 180, 320, 3])
            data = np.reshape(camera_observations, self.image_resolution[1])
        elif key == 'instruction':
            instruction_fp = os.path.join(demo_id, "instruction.zarr")
            # get proprio data
            data = zarr.load(instruction_fp)
            data = np.tile(data, (seq_end_index + 1 - seq_begin_index, 1))
        elif key == self.points_mask_key:
            points_mask_fp = os.path.join(demo_id, "{}.zarr".format(key))
            # get proprio data
            data = zarr.load(points_mask_fp)
            data = np.tile(data, (seq_end_index + 1 - seq_begin_index, 1))
        return data