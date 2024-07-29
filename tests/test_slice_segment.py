import torch
from src.tts.utils.audio import rand_slice_segments

def test_slice_segment():
    x = torch.ones((22, 192, 363))
    spec_lengths = torch.Tensor([
        304, 233, 192, 171, 343, 263, 248, 216,  92, 191, 
        226, 308, 289, 264, 216, 363, 196, 205, 228, 236, 270, 296
    ])
    segment_size = 128
    
    for _ in range(10):
        z_slice, ids_slice = rand_slice_segments(x, spec_lengths, segment_size)
        assert z_slice.size() == (22, 192, 128)
        assert ids_slice.size() == (22,)