import torch
from serotiny.networks._3d.unet import Unet

def test_unet():
    unet = Unet(in_channels=3, channel_fan=2, pooling='average')
    input = torch.zeros(1, 3, 160, 90, 60)
    output = unet(input)
    print(unet)
    print(f"UNET shape input: {input.shape} output: {output.shape}")

if __name__ == "__main__":
    test_unet()
