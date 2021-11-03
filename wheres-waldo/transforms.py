import random
import torch.nn.functional as F
import torchvision.transforms.functional as TF

class RandomCrop():
    def __init__(self, size):
        self.size = size

    def __call__(self, image, label, coord):
        w, h = image.size
        th, tw = self.size
        # Check if size is larger than image
        if w <= tw or h <= th:
            return image, label

        x1, y1 = 0, 0
        if random.random() < 0.5:
            # Do not (necessarily include Waldo)
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)
        else:
            # Include Waldo
            x1 = random.randint(coord[0] - tw, coord[0])
            y1 = random.randint(coord[1] - th, coord[1])

        return TF.crop(image, y1, x1, th, tw), TF.crop(label, y1, x1, th, tw), coord

class RandomScale():
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, image, label, coord):
        scale = random.uniform(self.scale[0], self.scale[1])    
        width, height = image.size
        
        return TF.resize(image, (int(height * scale), int(width * scale)), interpolation=TF.InterpolationMode.BILINEAR), \
            TF.resize(label, (int(height * scale), int(width * scale)), interpolation=TF.InterpolationMode.BILINEAR), \
                (int(scale * coord[0]), int(scale * coord[1]))


class ToTensor():
    def __call__(self, image, label, coord):
        return TF.to_tensor(image), TF.to_tensor(label), coord

class Compose():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label, coord):
        for t in self.transforms:
            image, label, coord = t(image, label, coord)
        return image, label, coord
