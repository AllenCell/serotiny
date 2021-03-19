import torch
from serotiny.networks._3d.unet import Unet

def test_unet():
    unet = Unet(depth=2, in_channels=3, channel_fan=2, out_channels=1, pooling='average', kernel_size=3, padding=1)
    #input = torch.zeros(1, 3, 160, 90, 60)
    input = torch.zeros(1, 3, 572, 572, 60)
    output = unet(input)
    #print(unet)
    print(f"UNET shape input: {input.shape} output: {output.shape}")

if __name__ == "__main__":
    test_unet()
