import os
import json

def replace_image_paths(root_path, old_prefix, new_prefix):
    """
    Traverse all subdirectories of `root_path`, and if an `images.json` is found,
    replace all image paths starting with `old_prefix` with `new_prefix`.
    """
    for dirpath, _, filenames in os.walk(root_path):
        if 'images.json' in filenames:
            json_path = os.path.join(dirpath, 'images.json')
            print(f"Processing {json_path}...")
            with open(json_path, 'r') as f:
                data = json.load(f)

            modified = False
            for key in data:
                new_list = []
                for path in data[key]:
                    if path.startswith(old_prefix):
                        new_path = path.replace(old_prefix, new_prefix, 1)
                        new_list.append(new_path)
                        modified = True
                    else:
                        new_list.append(path)
                data[key] = new_list

            if modified:
                with open(json_path, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"Updated {json_path}")
            else:
                print(f"No changes in {json_path}")

# === Usage ===
if __name__ == "__main__":
    root_episode_dir = "/datasets/llarva_v2/current/annotations/epic/epic_tasks/common_task"
    old_prefix = "/datasets/epic100_2024-01-04_1913/"
    new_prefix = "/datasets/epic100/current/"
    replace_image_paths(root_episode_dir, old_prefix, new_prefix)
