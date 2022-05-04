def info(path):
    from serotiny.io.image import image_loader

    img, channels = image_loader(path, return_channels=True, return_as_torch=False)
    print("Image shape: ", img.shape)
    print("Channel names: ", list(channels.keys()))
