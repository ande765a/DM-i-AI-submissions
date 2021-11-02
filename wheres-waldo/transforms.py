import random
import torch.nn.functional as F
import torchvision.transforms.functional as TF

class RandomCrop():
    def __init__(self, size):
        self.size = size

    def __call__(self, image, label):
        w, h = image.size
        th, tw = self.size
        # Check if size is larger than image
        if w <= tw or h <= th:
            return image, label
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return TF.crop(image, y1, x1, th, tw), TF.crop(label, y1, x1, th, tw)


class RandomRotate():
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, image, label):
        angle = random.uniform(-self.angle, self.angle)
        return TF.rotate(image, angle, mode="bilinear", align_corners=False), F.rotate(label, angle, mode="bilinear", align_corners=False)


class RandomScale():
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, image, label):
        scale = random.uniform(self.scale[0], self.scale[1])    
        width, height = image.size
        
        return TF.resize(image, (int(height * scale), int(width * scale)), interpolation=TF.InterpolationMode.BILINEAR), \
            TF.resize(label, (int(height * scale), int(width * scale)), interpolation=TF.InterpolationMode.BILINEAR)

class RandomHorizontalFlip():
    def __call__(self, image, label):
        if random.random() < 0.5:
            return TF.hflip(image), TF.hflip(label)
        return image, label

class ToTensor():
    def __call__(self, image, label):
        return TF.to_tensor(image), TF.to_tensor(label)

class Compose():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label
