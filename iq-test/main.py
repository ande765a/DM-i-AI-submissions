import numpy as np
import itertools
from PIL import Image
from binary_transforms import make_transforms
from unary_transforms import all_unary_transforms

#prefixes = ["1635866624843", "1635866624896", "1635866624972", "1635866625044", "1635866625131", "1635866625181", "1635866625254", "1635866625302", "1635866625376"]
prefixes = ["1635866625302"]

def get_tile(image, col, row):
    return image.crop((col * 2 * 110, row * 110, (col * 2 + 1) * 110, (row + 1) * 110))

if __name__ == "__main__":
  for prefix in prefixes:
    main_image = Image.open(f"images/{prefix}-image.jpg")
    image_choices = [Image.open(f"images/{prefix}-image-choice-{i}.jpg") for i in range(1, 4)]

    min_error = np.inf
    best_transform = None

    for row in range(0, 2):
      im1 = get_tile(main_image, 0, row)
      im2 = get_tile(main_image, 1, row)
      im3 = get_tile(main_image, 2, row)

      for make_transform in make_transforms:
        binary_transform = make_transform(im1, im2, im3)

        for unary_transforms in itertools.permutations(all_unary_transforms, r=2):
          error = 0
          for row in range(0, 2):
            im1 = get_tile(main_image, 0, row)
            im2 = get_tile(main_image, 1, row)
            im3 = get_tile(main_image, 2, row)

            pred = binary_transform(im1, im2)
            for unary_transform in unary_transforms:
              pred = unary_transform(pred)
            error += np.sum(np.square(np.array(pred) - np.array(im3)))

          if error < min_error:
            min_error = error
            best_transform = (binary_transform, unary_transforms)
          

    main_image.show()
    binary_transform, unary_transforms = best_transform

    pred = binary_transform(
      get_tile(main_image, 0, 3),
      get_tile(main_image, 1, 3)
    )

    for unary_transform in unary_transforms:
      pred = unary_transform(pred)
    
    min_error = np.inf
    best_choice = None
    for image_choice in image_choices:
      error = np.sum(np.square(np.array(pred) - np.array(image_choice)))
      if error < min_error:
        min_error = error
        best_choice = image_choice

    best_choice.show()
    input()