import numpy as np
from PIL import Image

def rotate_image_45_clockwise(image):
    """
    Rotate an image by 45 degrees clockwise.
    """
    return image.rotate(45)

def rotate_image_45_counterclockwise(image):
    """
    Rotate an image by 45 degrees counterclockwise.
    """
    return image.rotate(-45)

def rotate_image_90_clockwise(image):
    """
    Rotate an image by 90 degrees clockwise.
    """
    return image.rotate(90)


def rotate_image_90_counterclockwise(image):
    """
    Rotate an image by 90 degrees counterclockwise.
    """
    return image.rotate(-90)


def flip_image_horizontally(image):
    """
    Flip an image horizontally.
    """
    return image.transpose(Image.FLIP_LEFT_RIGHT)

def flip_image_vertically(image):
    """
    Flip an image vertically.
    """
    return image.transpose(Image.FLIP_TOP_BOTTOM)

def squeeze_horizontally(image):
    """
    Squeeze an image horizontally and keep background.
    """
    width, height = image.size
    new_im = Image.new('RGB', image.size, image.getpixel((0, 0)))
    new_width = int(width * 0.8) # Shrink 20%
    squeezed = image.resize((new_width, height))
    new_im.paste(squeezed, ((width - new_width) // 2, 0))
    return new_im

def squeeze_vertically(image):
    width, height = image.size
    new_im = Image.new('RGB', image.size, image.getpixel((0, 0)))
    new_height = int(height * 0.8) # Shrink 20%
    squeezed = image.resize((width, new_height))
    new_im.paste(squeezed, (0, (height - new_height) // 2))
    return new_im
    
def identity(image):
    """
    Return the image as is.
    """
    return image