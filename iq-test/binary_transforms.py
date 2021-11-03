import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

def average_image(image1, image2):
    """
    Returns the average of two images.
    """
    out = ((np.array(image1, dtype=np.uint16) + np.array(image2, dtype=np.uint16)) / 2).astype(np.uint8)
    return Image.fromarray(out)
    
def make_n_way_color_transform(image1, image2, image3, n):
  clusters = KMeans(n_clusters=5).fit(np.array(image1).reshape(-1, 3))
  im1 = KMeans.predict(clusters, np.array(image1).reshape(-1, 3))
  im2 = KMeans.predict(clusters, np.array(image2).reshape(-1, 3))
  im3 = KMeans.predict(clusters, np.array(image3).reshape(-1, 3))

  transforms = []

  for i in range(0, 5):
    for j in range(0, 5):
      for k in range(0, 5):
        if i == j == k:
          continue

        c = (im3[(im1 == i) * (im2 == j)] == k).sum() # How many is satisfied.
        if c > 0:
          transforms.append((i, j, k, c))
        
  transforms = sorted(transforms, key=lambda x: x[3], reverse=True)[:n]

  def transform(image1, image2):
    im1 = KMeans.predict(clusters, np.array(image1).reshape(-1, 3))
    im2 = KMeans.predict(clusters, np.array(image2).reshape(-1, 3))

    im3 = im2.copy()

    for i, j, k, _ in transforms:
      im3[(im1 == i) * (im2 == j)] = k

    im3 = np.array([clusters.cluster_centers_[k] for k in im3])
    im3 = im3.astype(np.uint8).reshape((*image1.size, 3))

    return Image.fromarray(im3)

  return transform


def make_3_way_color_transform(image1, image2, image3):
  return make_n_way_color_transform(image1, image2, image3, 3)

def make_2_way_color_transform(image1, image2, image3):
  return make_n_way_color_transform(image1, image2, image3, 2)

def make_identity_transform(image1, image2, image3):
  return lambda image1, image2: image2