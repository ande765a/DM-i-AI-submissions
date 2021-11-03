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


def identity(image):
    """
    Return the image as is.
    """
    return image