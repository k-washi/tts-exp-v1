import torch
from pathlib import Path
from src.config.config import Config
from src.models.vits.plmodule import ViTSModule
from src.models.vits.infer import VitsInfer


cfg = Config()

def plmodule_to_pytorch_module(checkpoint, output_path):
    model = ViTSModule.load_from_checkpoint(checkpoint, cfg=cfg)
    torch.save(model.net_g.state_dict(), output_path)
    
    vits = VitsInfer(cfg.model.net_g, cfg.data)
    vits.load_state_dict(torch.load(output_path))
    vits.eval()
    output_path = output_path.replace(".pth", ".onnx")
    text_length = 37
    text_phones = torch.randint(0, 42, (1, text_length)).long()
    accent_padded = torch.randint(0, 3, (1, text_length)).long()
    sid = torch.tensor([0]).long()
    
    device = "cpu"
    input_names = ["text_padded", "accent_pos_padded", "speaker_id"]
    output_names = ["audio"]
    
    torch.onnx.export(
        vits,
        (
            text_phones.to(device),
            accent_padded.to(device),
            sid.to(device)
        ),
        output_path,
        dynamic_axes={
            "text_padded": {0: "batch", 1: "text_length"},
            "accent_pos_padded": {0: "batch", 1: "text_length"},
            "speaker_id": {0: "batch"},
            "audio": {0: "batch", 1: "audio_channel", 2: "audio_length"}
        },
        do_constant_folding=False,
        opset_version=13,
        verbose=False,
        input_names=input_names,
        output_names=output_names
        
    )
    print(output_path)
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="logs/00013/checkpoints/checkpoint-epoch=22399-val/total_loss=27.7434.ckpt")
    parser.add_argument("--output_path", type=str, default="logs/00013/net_g_c_sayoko_vits_ep22399.pth")
    
    args = parser.parse_args()
    plmodule_to_pytorch_module(args.checkpoint, args.output_path)
    

    
    