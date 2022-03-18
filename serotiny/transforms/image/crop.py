from .pad import PadTo


class CropCenter:
    def __init__(
        self, cropz, cropy, cropx, pad=0, center_of_mass=None, force_size=True
    ):
        self.cropz = cropz + (cropz % 2 != 0)
        self.cropy = cropy + (cropy % 2 != 0)
        self.cropx = cropx + (cropx % 2 != 0)

        self.pad = pad
        self.center_of_mass = center_of_mass
        self.force_size = force_size

    def __call__(self, img):
        c, z, y, x = img.shape

        if self.center_of_mass is None:
            center_of_mass = (z // 2, y // 2, x // 2)
        else:
            center_of_mass = self.center_of_mass

        startz = max(0, center_of_mass[0] - (self.cropz // 2) - self.pad)
        starty = max(0, center_of_mass[1] - (self.cropy // 2) - self.pad)
        startx = max(0, center_of_mass[2] - (self.cropx // 2) - self.pad)

        endz = startz + self.cropz + 2 * self.pad
        endy = starty + self.cropy + 2 * self.pad
        endx = startx + self.cropx + 2 * self.pad

        img = img[:, startz:endz, starty:endy, startx:endx]

        if self.force_size:
            pad_to = PadTo(target_dims=[self.cropz, self.cropy, self.cropx])
            img = pad_to(img)

        return img
