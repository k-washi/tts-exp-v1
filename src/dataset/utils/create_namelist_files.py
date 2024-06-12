from pathlib import Path
import math
def create_namelist_files(input_dir: str, output_dir: str, val_rate: float):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    files = sorted(list(input_dir.glob("*")))
    name_list = [f.stem for f in files]
    train_num = len(files) - math.floor(len(files) * val_rate)
    train_list, val_list = name_list[:train_num], name_list[train_num:]
    
    save_list(train_list, str(output_dir / "train.txt"))
    save_list(val_list, str(output_dir / "val.txt"))

def save_list(name_list: list, output_path: str):
    s = "\n".join(name_list)
    with open(output_path, "w", encoding='utf-8') as f:
        f.write(s)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--val_rate", type=float, default=0.02)
    
    args = parser.parse_args()
    create_namelist_files(args.input, args.output, args.val_rate)