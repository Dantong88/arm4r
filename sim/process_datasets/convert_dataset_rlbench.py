import pickle
from PIL import Image
from scipy.spatial.transform import Rotation as R
import argparse
import os
from utils import *
import h5py
import numpy as np
from tqdm import tqdm
import zarr
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

def img2bytes(raw_image):
    return raw_image.tobytes()  # Simulate the raw byte format


def get_curr_pos(item):
    gripper_pos = getattr(item, "gripper_pose")
    gripper_open = getattr(item, "gripper_open")
    quat_rot = gripper_pos[-4:]

    euler = R.from_quat(quat_rot).as_euler("XYZ")
    gripper_position = np.array([gripper_open], dtype=np.float32).reshape(-1, 1)


    # here calculate the action for action head return
    # rot = quat_to_rot_6d(quat_rot.reshape(1, -1)) # contruct the 6d rot
    cartesian_position = np.concatenate([gripper_pos[:3].reshape(1, -1), euler.reshape(1, -1)], axis=1, dtype=np.float32)
    return cartesian_position, gripper_position


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



def img2arr(image_path):
    try:
        img_pil = Image.open(image_path)
        img_arr = np.array(img_pil)
    except:
        img_arr = np.random.randint(0, 255, (128,128,3), dtype="uint8")


    return img_arr

def save_arr_dict(data, out_zarr_path: str):
    zarr.save(out_zarr_path, data)

def create_hdf5(file_path, episodes_data, side_image_size=49152):
    """
    Create an HDF5 file with episodes stored in the required format.

    Parameters:
        file_path (str): Path to save the HDF5 file.
        episodes_data (dict): Dictionary with episode data.
        side_image_size (int): Size of side images in bytes (e.g., 172800).
    """
    with h5py.File(file_path, "w") as hdf5_file:
        for episode_name, data in episodes_data.items():
            # Create group for the episode
            group = hdf5_file.create_group(episode_name)

            # Create datasets directly at the expected paths
            group.create_dataset(
                "observation/exterior_image_1_left",
                data=np.array(data["side_images"], dtype=f"|S{side_image_size}"),
                compression="gzip",
            )
            group.create_dataset(
                "observation/wrist_image_left",
                data=np.array(data["wrist_images"], dtype=f"|S{side_image_size}"),
                compression="gzip",
            )

            # Add action data
            for key, values in data["actions"].items():
                group.create_dataset(f"action/{key}", data=values, compression="gzip")

            # Add proprioceptive data
            for key, values in data["proprios"].items():
                group.create_dataset(f"observation/{key}", data=values, compression="gzip")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process rlbench, the input should be python dict including episodes')
    parser.add_argument('--eps_list', type=str,
                        default='/home/niudt/project/arm4r_release/arm4r/sim/process_datasets/annotations/train/meat_off_grill_400/400eps.json')
    parser.add_argument('--save_root', type=str,
                        default='annotations/arm4r_format')
    parser.add_argument('--task', type=str,
                        default='meat_off_grill_400_front_wrist')
    parser.add_argument('--primitive', type=str,
                        default='rlbench_tasks')

    parser.add_argument('--camera_view', type=str, nargs='+', default=['front_rgb', 'wrist_rgb'],
                        help='List of tasks to process (space-separated)')

    args = parser.parse_args()

    text_tokenizer = Text_Tokenizer()


    # load the processing list
    eps_list = json.load(open(args.eps_list))


    ann_dict = {}
    eps_count = 0
    for task in eps_list:
        for eps in tqdm(eps_list[task]):
            current_episode_path = eps


            variation_des_pickle = os.path.join(current_episode_path, "variation_descriptions.pkl")
            with open(variation_des_pickle, 'rb') as file:
                try:
                    variation_des = pickle.load(file)
                except:
                    continue

            pickle_path = os.path.join(current_episode_path, "low_dim_obs.pkl")

            with open(pickle_path, 'rb') as file:
                current_eps_info = []
                current_eps_info_gt = []
                instruction = 'The task is \"{}\".'.format(variation_des[0])
                data = pickle.load(file)

                # extract frame
                # save the proprios information
                cartesian_positions = []
                gripper_positions = []
                cartesian_positions_cmd = []
                gripper_positions_cmd = []
                exterior_image_1_left = []
                wrist_image_left = []
                img1_test = []
                img2_test = []
                for idx in range(len(data) - 1):
                    cartesian_position, gripper_position = get_curr_pos(data[idx])
                    cartesian_positions.append(cartesian_position)
                    gripper_positions.append(gripper_position)

                    cartesian_position_cmd, gripper_position_cmd = get_curr_pos(data[idx + 1])
                    cartesian_positions_cmd.append(cartesian_position_cmd)
                    gripper_positions_cmd.append(gripper_position_cmd)

                    exterior_image_1_left_path = '{}/{}/{}.png'.format(current_episode_path, args.camera_view[0], idx)
                    try:
                        img1 = Image.open(exterior_image_1_left_path)
                    except:
                        img1 = np.zeros([128,128,3])
                    wrist_image_left_path = '{}/{}/{}.png'.format(current_episode_path, args.camera_view[1], idx)
                    try:
                        img2 = Image.open(wrist_image_left_path)
                    except:
                        img2 = np.zeros([128, 128, 3])

                    img1_test.append(np.asarray(img1))
                    img2_test.append(np.asarray(img2))

                    # exterior_image_arr = img2arr(exterior_image_1_left_path)
                    exterior_image_1_left.append(exterior_image_1_left_path)
                    # wrist_image_left_arr = img2arr(wrist_image_left_path)
                    wrist_image_left.append(wrist_image_left_path)

                try:
                    assert np.stack(img1_test, axis=0).shape[1:] == (128, 128, 3)
                    assert np.stack(img2_test, axis=0).shape[1:] == (128, 128, 3)
                except:
                    print(current_episode_path)

                cartesian_positions = np.concatenate(cartesian_positions, axis=0)
                gripper_positions = np.concatenate(gripper_positions, axis=0)
                cartesian_positions_cmd = np.concatenate(cartesian_positions_cmd, axis=0)
                gripper_positions_cmd = np.concatenate(gripper_positions_cmd, axis=0)
                proprio = np.concatenate([cartesian_positions, gripper_positions], axis=-1)
                action = np.concatenate([cartesian_positions_cmd, gripper_positions_cmd], axis=-1)
                instruction = text_tokenizer.tokenize(variation_des[0])[0].numpy()

                task_dir = os.path.join(args.save_root, args.task, args.primitive, 'common_task')
                proprio_fp = os.path.join(task_dir, '%05d'%eps_count, "proprio.zarr")
                action_fp = os.path.join(task_dir, '%05d'%eps_count, "action.zarr")
                image_fp = os.path.join(task_dir, '%05d'%eps_count, "images.json")
                instruction_fp = os.path.join(task_dir, '%05d'%eps_count, "instruction.zarr")

                save_arr_dict(proprio, proprio_fp)
                save_arr_dict(action, action_fp)
                save_arr_dict(instruction, instruction_fp)
                image_info = {}
                image_info['observation/exterior_image_1_left'] = exterior_image_1_left
                image_info['observation/wrist_image_left'] = wrist_image_left

                with open(image_fp, "w") as json_file:
                    json.dump(image_info, json_file)

                eps_count += 1


