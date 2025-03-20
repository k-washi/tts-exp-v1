import torch
from pathlib import Path
from src.config.config import Config, get_config
from src.models.msistft.plmodule_cn import MsISTFTModule
from src.models.msistft.infer import MsISTFTInfer


cfg = get_config()
print(cfg)

cfg.dataset.add_blank_type = 2 # 0: なし, 1: 音素の前後に挿入, 2: モーラの前後に挿入
cfg.dataset.accent_split = True # アクセントを分割するか
cfg.dataset.accent_up_ignore = False # アクセント上昇を無視するか
cfg.model.net_g.use_noise_scaled_mas = False
cfg.model.net_g.mas_nosie_scale_initial = 0.01
cfg.model.net_g.mas_noise_scale_delta = 5e-7
cfg.model.net_g.flow_n_resblocks = 4


def plmodule_to_pytorch_module(checkpoint, output_path):
    #model = MsISTFTModule.load_from_checkpoint(checkpoint, cfg=cfg)
    model = MsISTFTModule(cfg=cfg)
    model.net_g.load_state_dict(torch.load(checkpoint))
    torch.save(model.net_g.state_dict(), output_path)
    
    vits = MsISTFTInfer(cfg.model.net_g, cfg.dataset)
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
    parser.add_argument("--checkpoint", type=str, default="logs/msistft_accent_mblank_noduradv_00402_noutomi_ft/ckpt/ckpt-3000/net_g.pth")
    parser.add_argument("--output_path", type=str, default="logs/msistft_accent_mblank_noduradv_00402_noutomi_ft/net_g_c_noutomi_smistft_ep3000.pth")
    
    args = parser.parse_args()
    plmodule_to_pytorch_module(args.checkpoint, args.output_path)
    

    
    