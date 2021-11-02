import os
from PIL import Image
import matplotlib.pyplot as plt



if __name__ == "__main__":
  images_path = os.path.join("images")
  image_filenames = os.listdir(images_path)
  
  for image_filename in image_filenames:
    if image_filename == ".DS_Store":
      continue

    image_path = os.path.join(images_path, image_filename)

    image = Image.open(image_path)
    
    plt.title(image_filename)
    plt.imshow(image)
    plt.show()
      



