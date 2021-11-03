import random
import torch.nn.functional as F
import torchvision.transforms.functional as TF

class RandomCrop():
    def __init__(self, size):
        self.size = size

    def __call__(self, image, label, coord, visible):
        w, h = image.size
        th, tw = self.size
        # Check if size is larger than image
        if w <= tw or h <= th:
            return image, label, coord, visible

        x1, y1 = 0, 0
        if random.random() < 0.5:
            # Do not (necessarily include Waldo)
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)
            visible = 1 if x1 <= coord[0] and coord[0] <= x1 + tw and y1 <= coord[1] and coord[1] <= y1 + th else 0
        else:
            # Include Waldo
            x1 = random.randint(coord[0] - tw, coord[0])
            y1 = random.randint(coord[1] - th, coord[1])

        return TF.crop(image, y1, x1, th, tw), TF.crop(label, y1, x1, th, tw), (coord[0] - x1, coord[1] - y1), visible

class RandomScale():
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, image, label, coord, visible):
        scale = random.uniform(self.scale[0], self.scale[1])    
        width, height = image.size
        
        return  TF.resize(image, (int(height * scale), int(width * scale)), interpolation=TF.InterpolationMode.BILINEAR), \
                TF.resize(label, (int(height * scale), int(width * scale)), interpolation=TF.InterpolationMode.BILINEAR), \
                (int(scale * coord[0]), int(scale * coord[1])), \
                visible


class ToTensor():
    def __call__(self, image, label, coord, visible):
        return TF.to_tensor(image), TF.to_tensor(label), coord, visible

class Compose():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label, coord, visible):
        for t in self.transforms:
            image, label, coord, visible = t(image, label, coord, visible)
        return image, label, coord, visible
