import argparse
import json
from pathlib import Path
from tqdm import tqdm

def update_paths_in_json(json_path, prefix):
    with open(json_path, 'r') as f:
        data = json.load(f)

    updated = {}
    for key, paths in data.items():
        updated_paths = []
        for p in paths:
            parts = Path(p).parts
            if "frames" not in parts:
                raise ValueError(f"'frames' not found in path: {p}")
            idx = parts.index("frames")
            new_p = str(Path(prefix) / Path(*parts[idx:]))
            updated_paths.append(new_p)
        updated[key] = updated_paths

    with open(json_path, 'w') as f:
        json.dump(updated, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Update image paths in images.json files.")
    parser.add_argument("--root_dir", default="path_to/epic_tasks_final/common_task", help="Directory containing episode folders like 000000, 000001, etc.")
    parser.add_argument("--prefix", default="epic_kitchens_data", help="New prefix to apply before 'frames'")
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    episode_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir()])

    for episode_dir in tqdm(episode_dirs, desc="Processing episodes"):
        json_file = episode_dir / "images.json"
        if not json_file.exists():
            raise FileNotFoundError(f"'images.json' not found in: {episode_dir}")
        update_paths_in_json(json_file, args.prefix)

if __name__ == "__main__":
    main()
