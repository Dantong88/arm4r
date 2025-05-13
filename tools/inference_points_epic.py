import json
import  os
import h5py
import numpy as np
from arm4r.models.policy.arm4r_wrapper import ARM4RWrapper_pretrain_single_view
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import trange
from IPython.display import HTML
import zarr
import argparse

def load_points(epi, s, e, num_points, points_key):
    file = os.path.join(epi, '{}.zarr'.format(points_key))
    points = zarr.load(file)[s:e + 1, :num_points]
    standard_points = np.zeros([points.shape[0], num_points, 3])
    standard_points[:, :points.shape[1], :] = points
    standard_points[:, :, 0] *= 0.002
    standard_points[:, :, 1] *= 0.002
    return standard_points

def get_data_from_demo_path(demo_path, length, image_resolution=[[-1,256,456,3], [-1, 256,456,3]], observation_key = ['observation/exterior_image_1_left', 'observation/wrist_image_left'], points_key = 'wrist_points',return_PIL_images=False):
    camera_observations = {}
    for key in observation_key:
        camera_observations[key] = []
        for img_dix in range(length):
            try:
                if os.path.exists(os.path.join(demo_path, 'images')):
                    current_frame_path = os.path.join(demo_path, 'images', os.path.basename(key), '%05d.jpg' % img_dix)
                    current_frame = Image.open(current_frame_path)
                else:
                    image_json = json.load(open(os.path.join(demo_path, 'images.json')))
                    current_frame_path = image_json[key][img_dix]
                    current_frame = Image.open(current_frame_path)
            except:
                current_frame = np.random.randint(0, 255, image_resolution[0][1:], dtype="uint8")
            if not current_frame.size == (image_resolution[0][2], image_resolution[0][1]):
                current_frame = current_frame.resize((image_resolution[0][2], image_resolution[0][1]))
            camera_observations[key].append(np.asarray(current_frame))
        camera_observations[key] = np.stack(camera_observations[key])

    # img, action and proprio keys can be found in config/dataset_config_template.yaml
    side_images_l = camera_observations[observation_key[0]]
    wrist_images_l = camera_observations[observation_key[1]]

    proprios = load_points(demo_path, 0, length - 1, 1296, points_key).reshape([length, -1])
    actions = load_points(demo_path, 1, length, 1296, points_key).reshape([length, -1])

    instruction = zarr.load(os.path.join(demo_path, 'instruction.zarr'))

    if return_PIL_images:
        side_images_l = [Image.fromarray(side_images_l[i]) for i in range(side_images_l.shape[0])]
        wrist_images_l = [Image.fromarray(wrist_images_l[i]) for i in range(wrist_images_l.shape[0])]

    return {
        "side_images": side_images_l,
        "wrist_images": wrist_images_l,
        "actions": actions,
        "proprios": proprios,
        "instruction": instruction
    }

def decode_points(pred, x_weights = 0.002, y_weights = 0.002, z_weights = 1):
    front_preds = np.reshape(pred, [-1, 3])
    front_preds[:, 0] /= x_weights
    front_preds[:, 1] /= y_weights
    front_preds[:, 2] /= z_weights

    return front_preds


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference of the predition of the points')
    parser.add_argument('--demo_path', type=str,
                        default='PREFIX/epic/epic_tasks/common_task/022082') # choose a random demo from the dataset we release
    parser.add_argument('--ckpt', type=str,
                        default='../arm4r-ckpts/model_ckpts/pretrained_epic.pth') # use our released ckpt
    parser.add_argument('--points_key', type=str,
                        default='points')
    parser.add_argument('--image_resolution', nargs = '+', type = str, default=[[-1,256,456,3], [-1, 256,456,3]])
    parser.add_argument('--observation_key', nargs='+', type=str, default=["observation/ego_image","observation/ego_image"])

    args = parser.parse_args()


    checkpoint_path = args.ckpt
    train_yaml_path = os.path.join(os.path.dirname(args.ckpt), "run.yaml")
    vision_encoder_path = "../arm4r-ckpts/vision_encoder/cross-mae-rtx-vitb.pth"


    arm4r = ARM4RWrapper_pretrain_single_view(train_yaml_path, checkpoint_path, vision_encoder_path)


    ## choose eps
    demo_path = args.demo_path

    ## get_length
    length = zarr.load(os.path.join(demo_path, '{}.zarr').format(args.points_key)).shape[0] - 1



    obs_dict = get_data_from_demo_path(demo_path=demo_path, length=length, image_resolution=args.image_resolution,
                                       observation_key=args.observation_key, points_key=args.points_key)
    side_images_l, wrist_images_l, proprios, actions, instruction = obs_dict["side_images"], obs_dict["wrist_images"], \
    obs_dict["proprios"], obs_dict["actions"], obs_dict["instruction"]

    frames = np.concatenate([side_images_l, wrist_images_l], axis=1)
    T = frames.shape[0]

    fig, ax = plt.subplots()
    img = ax.imshow(frames[0])

    def update(frame):
        img.set_data(frame)
        return [img]

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=50, blit=True)


    side_images = [Image.fromarray(side_images_l[i]) for i in range(side_images_l.shape[0])]
    wrist_images = [Image.fromarray(wrist_images_l[i]) for i in range(wrist_images_l.shape[0])]

    arm4r.reset()
    pred_points_eps = []
    for i in trange(len(side_images)):
        if i == 0:
            action = None
            # get the first frame points for to save for later reconstruction
            first_front = proprios[0].copy()
            pred_points = decode_points(first_front, x_weights=0.002, y_weights=0.002, z_weights=1)
            pred_points_eps.append(pred_points)
        else:
            action = actions[i-1:i]
        action = arm4r(
            side_images[i], wrist_images[i],
            proprios[i:i+1],
            instruction=instruction,
            action=action,
            use_temporal=False,
            teacher_forcing=False,
            binary_gripper=False,
        )
        pred_points = decode_points(action, x_weights = 0.002, y_weights = 0.002, z_weights = 1)
        # save_points
        pred_points_eps.append(pred_points )

        gt_points = zarr.load(os.path.join(demo_path, '{}.zarr').format(args.points_key))[i+1]
        print('*' * 50)
        print('step: {}'.format(i))
        print('difference: ', abs(pred_points - gt_points).mean())
    pred_points_ep = np.stack(pred_points_eps, axis=0)

    # save the pred points
    pred_save_dir = os.path.join(args.demo_path, 'pred_{}.npy'.format(args.points_key))
    os.makedirs(os.path.dirname(pred_save_dir), exist_ok=True)
    np.save(pred_save_dir, pred_points_ep)
