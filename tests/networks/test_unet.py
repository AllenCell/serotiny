import torch

#from serotiny.networks._3d.unet_old import Unet_Old
from serotiny.networks._3d.unet import Unet

def test_unet():
    #input = torch.zeros(1, 3, 160, 90, 60)
    #input = torch.zeros(1, 3, 572, 572, 60)
    input = torch.rand(1, 3, 572, 572, 60)
    
    #unet = Unet_Old(depth=2, in_channels=3, channel_fan=2, out_channels=1, pooling='average', kernel_size=3, padding=1)
    unet = Unet(depth=2, in_channels=3, channel_fan=2, out_channels=1, pooling='average', kernel_size=3, padding=1)
    
    output = unet(input)
    
    #print(unet)
    
    '''
    print('networks_down:')
    unet.print_network(unet.networks_down, reverse=True)
    print()

    print('networks_up:')
    unet.print_network(unet.networks_up, reverse=False)
    print()
    
    print('conv_out:')
    print(unet.conv_out)
    print()
    '''
    
    unet.print_network()
    
    print(f"UNET shape input: {input.shape} output: {output.shape}")

if __name__ == "__main__":
    test_unet()
