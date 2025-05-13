import os
import json
import pickle
import argparse
import random
from tqdm import tqdm

def traverse_path(info_dict, path, key):
    tasks = os.listdir(path)
    s = 1
    for task in tasks:
        if 'zip' in task:
            continue
        if key in task:
            eps_list = [os.path.join(path, task, 'all_variations/episodes/', i) for i in
                        os.listdir(os.path.join(path, task, 'all_variations/episodes/'))]
            info_dict[key] += eps_list
            s = 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Traverse all available data and select n eps for each vari')
    parser.add_argument('--save_all', type=bool,
                        default=False)
    parser.add_argument('--num_per_vari', type=int,
                        default=200)
    parser.add_argument('--traverse_paths', nargs='+', type=str,
                        default=['../data/train'])
    parser.add_argument('--work_dir', type=str,
                        default='annotations/')
    parser.add_argument('--save_name', type=str,
                        default='train')
    parser.add_argument('--task', type=str,
                        default='meat_off_grill')
    parser.add_argument('--val', type=bool,
                        default=False)

    parser.add_argument('--standard_val', type=bool,
                        default=False)

    parser.add_argument('--combined_task', type=bool,
                        default=True)

    parser.add_argument('--upper_bound', type=int,
                        default=400)

    args = parser.parse_args()

    info_dict = {}
    current_task = args.task
    info_dict[current_task] = []
    for path in args.traverse_paths:
        traverse_path(info_dict, path, current_task)

    # this is find the variations
    info_dict_vari = {}
    for task in info_dict:
        for eps in info_dict[task]:
            variation_des_pickle = os.path.join(eps, "variation_descriptions.pkl")
            with open(variation_des_pickle, 'rb') as file:
                try:
                    variation_des = pickle.load(file)[0]
                except:
                    continue
                if not variation_des in info_dict_vari:
                    info_dict_vari[variation_des] = []
                if not args.save_all:
                    if len(info_dict_vari[variation_des]) < args.num_per_vari:
                        info_dict_vari[variation_des].append(eps)


    if args.combined_task:
        combined_info_dict_vari = {'rlbench_tasks': []}
        for key in info_dict_vari:
            combined_info_dict_vari['rlbench_tasks'] += info_dict_vari[key]
        random.shuffle(combined_info_dict_vari['rlbench_tasks'])
        combined_info_dict_vari['rlbench_tasks'] = combined_info_dict_vari['rlbench_tasks'][:args.upper_bound]
        total_num_eps = len(combined_info_dict_vari['rlbench_tasks'])
        info_dict_vari = combined_info_dict_vari
    if args.save_all:
        save_path = os.path.join(args.work_dir, args.save_name, 'all_{}.json'.format(total_num_eps))
    save_path = os.path.join(args.work_dir, args.save_name, current_task + '_{}'.format(total_num_eps),
                             '{}eps.json'.format( total_num_eps))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w') as json_file:
        json.dump(info_dict_vari, json_file)
