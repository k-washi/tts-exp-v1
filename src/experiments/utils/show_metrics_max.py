import numpy as np
from pathlib import Path

def show_epoch_sort_by_metric(metric_list:list, epoch_dirs:list, show_num:int=5):
    sorted_epoch_dirs, sorted_metrics = zip(*sorted(zip(epoch_dirs, metric_list), key=lambda x: x[1], reverse=True))
    for epoch_dir, metric in zip(sorted_epoch_dirs[:show_num], sorted_metrics[:show_num]):
        print(f"{epoch_dir}: {metric}")
    
    return sorted_epoch_dirs, sorted_metrics

def show_metrics_max(target_dir:str, show_num:int=5):
    target_dir = Path(target_dir)
    epoch_dirs = sorted(list(target_dir.glob("*")))
    epoch_list = [str(epoch_dir.stem) for epoch_dir in epoch_dirs]
    xvector_sim_list, speech_mos_list, speech_bert_score_f1_list = [], [], []
    
    for epoch_dir in epoch_dirs:
        res_file = epoch_dir / "result.txt"
        if not res_file.exists():
            print(f"Result file not found: {res_file}")
            continue
        with open(res_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                ref_name = line.split(":")[0]
                if "xvector_sim" == ref_name:
                    xvector_sim = float(line.split(":")[-1].replace(" ", ""))
                    xvector_sim_list.append(xvector_sim)
                if "gen_mos" == ref_name:
                    speech_mos = float(line.split(":")[-1].replace(" ", ""))
                    speech_mos_list.append(speech_mos)
                if "speech_bert_score_f1" == ref_name:
                    speech_bert_score_f1 = float(line.split(":")[-1].replace(" ", ""))
                    speech_bert_score_f1_list.append(speech_bert_score_f1)
    
    print(f"Xvector Sim Max: {max(xvector_sim_list)}")
    print(f"Speech MOS Max: {max(speech_mos_list)}")
    print(f"Speech Bert Score F1 Max: {max(speech_bert_score_f1_list)}")

    print("Xvector Sim", "-"*50)
    sorted_epoch_dirs, sorted_metrics = show_epoch_sort_by_metric(xvector_sim_list, epoch_list)
    
    print("Speech MOS", "-"*50)
    sorted_epoch_dirs, sorted_metrics = show_epoch_sort_by_metric(speech_mos_list, epoch_list)
    
    print("Speech Bert Score F1", "-"*50)
    sorted_epoch_dirs, sorted_metrics = show_epoch_sort_by_metric(speech_bert_score_f1_list, epoch_list)
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Show metrics max')
    parser.add_argument('--target_dir', type=str, default='logs/vits_base_00002/val')
    
    show_metrics_max(**vars(parser.parse_args()))