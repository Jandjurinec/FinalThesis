import numpy as np
import cv2
from tqdm import tqdm
from scipy.ndimage import convolve
from scipy.stats import entropy
from skimage.filters import sobel
import imutils

np.random.seed(101)

LIMITS = [[0, 0.8],
          [-1, 1],
          [0, 0.3],
          [0.5, 1.5]]

VELOCITY_LIMIT = 1

class PSOptimizer:
  def __init__(self, img, w, c1, c2, num_particles=20, is_global_neighborhood=True, kernel_size=3):
    self.img = img
    self.kernel_size = kernel_size
    self.w = w
    self.c1 = c1
    self.c2 = c2
    self.num_particles = num_particles
    self.is_global_neighborhood = is_global_neighborhood

    self.population = [np.random.uniform(0, 0.5, 4) for _ in range(num_particles)]
    self.velocities = [np.random.uniform(-VELOCITY_LIMIT, VELOCITY_LIMIT, 4) for _ in range(num_particles)]

    self.personal_best = np.copy(self.population)
    self.personal_best_fitness = np.zeros(num_particles)

    self.global_best = None
    self.global_best_fitness = 0

    self.fimg = None

    self._evaluate_particles()

  def _evaluate_particles(self):
    for i, particle in enumerate(self.population):
      img , fitness = img_enhance(self.img, particle[0], particle[1], particle[2], particle[3], n=self.kernel_size)
      if fitness > self.personal_best_fitness[i]:
        self.personal_best[i] = particle
        self.personal_best_fitness[i] = fitness
      if fitness > self.global_best_fitness:
        self.global_best = particle
        self.global_best_fitness = fitness
        self.fimg = img

  def step(self):
    for i in range(len(self.population)):
      p = self.population[i]
      v = self.velocities[i]
      p_best = self.personal_best[i]
      g_best = self.global_best

      r1 = np.random.rand(*p.shape)
      r2 = np.random.rand(*p.shape)

      cognitive = self.c1 * r1 * (p_best - p)
      social = None
      if self.is_global_neighborhood:
        social = self.c2 * r2 * (g_best - p) # global neigh
      else:
        # chain topology
        left, right = None, None
        if i == 0:
          left = 0
          right = i+1
        elif i == len(self.population)-1:
          left = i-1
          right = 0
        else:
          left = i-1
          right = i+1
        if self.personal_best_fitness[left] > self.personal_best_fitness[right]:
          social = self.c2 * r2 * (self.personal_best[left] - p)
        else:
          social = self.c2 * r2 * (self.personal_best[right] - p)

      new_v = self.w * v + cognitive + social
      new_p = p + new_v

      # check v and p boundries
      for r in range(4):
        new_p[r] = np.clip(new_p[r], LIMITS[r][0], LIMITS[r][1])
        new_v[r] = np.clip(new_v[r], -VELOCITY_LIMIT, VELOCITY_LIMIT)

      self.velocities[i] = new_v
      self.population[i] = new_p
      self._evaluate_particles()

def img_enhance(img, a=0.1, b=0.2, c=0.3, k=1.5, n=121):
  '''
    Function to enhance the original image.
      f(i,j) is an original image
      m(i,j) is the local mean of (i,j)th pixel
      sigm(i,j) is the standard deviation of (i,j)th pixel
      D is a global mean over the whole image

      g(i,j) = [k*D / (sigm(i,j) + b)] * [f(i,j) - c*m(i,j)] + m(i,j) ** a
      g(i,j) is an enhanced image
  '''
  # get local mean of (i,j)
  kernel = np.ones((n, n))
  local_mean = convolve(img, kernel, mode='reflect')

  # get local std of (i,j)
  std = pow(np.sum(abs(img - local_mean) ** 2) / (img.shape[0]*img.shape[1] - 1), 0.5)

  # get global mean of pixels within MxN
  D = np.mean(img)

  # final result
  g = (k * (D / (std + b))) * (img - c * local_mean) + local_mean ** a

  # calculate fitness score
  hist, _ = np.histogram(g, bins=256, range=(0, 256), density=True)
  hist = hist[hist > 0]
  image_entropy = entropy(hist, base=2)
  sobel_edges = sobel(g)
  edge_intensity = np.sum(sobel_edges)
  edges_count = np.sum(sobel_edges > 0.1)
  fitness = np.log(image_entropy * edge_intensity * edges_count)

  return g, fitness

def fill_holes(img):
  '''
    Fill small holes in a binary image using morphological opening.
  '''
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(19,19))
  return cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)

def strip_skull_PSO(dataset, K_SIZE=3):
  '''
    Removes skull from MRI images using Particle Swarm Optimization (PSO) and connected component analysis.

    This method processes a dataset of MRI images to differentiate between brain tissue and skull 
    by applying a PSO-based optimization technique. It normalizes each image, converts it to grayscale,
    and utilizes PSO to enhance the brain region. The largest connected component is identified 
    as the brain, and a mask is created to separate the brain tissue from the skull.

    Parameters:
      dataset : numpy.ndarray
    
      K_SIZE : int, optional
        The size of the kernel used in the PSO optimization process. Default is 3.

    Returns:
      numpy.ndarray
        A new dataset with the skull stripped from each MRI image. The output has the same shape 
        as the input dataset but contains only the brain tissue or background based on the analysis.

    Notes:
    - The method assumes that the input dataset contains a mix of T1 and T2 weighted MR scans.
    - The function uses OpenCV for image processing tasks such as normalization, thresholding, 
      and connected components analysis.
    - The final decision on whether to keep the foreground or background is made based on 
      the proportion of black pixels in each.
  '''
  new_dataset = np.copy(dataset)
  for i, X in enumerate(tqdm(dataset)):
    normalized = cv2.normalize(X, None, 0, 255, cv2.NORM_MINMAX)
    img = normalized.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pso_optimizer = PSOptimizer(img, w=0.6, c1=2.05, c2=1.7, num_particles=20, is_global_neighborhood=True, kernel_size=K_SIZE)
    for _ in range(3):
      pso_optimizer.step()
    fimg = pso_optimizer.fimg
    fimg = cv2.normalize(fimg, None, 0, 255, cv2.NORM_MINMAX)
    fimg = fimg.astype(np.uint8)

    _, thresh = cv2.threshold(fimg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    brain_mask = np.where(labels == largest, 0, 1).astype('uint8')
    blurred_mask = fill_holes(cv2.medianBlur(brain_mask, ksize=3))

    # because dataset is a mix of T1 and T2 weighted MR scans
    foreground = X.copy()
    background = X.copy()

    background[blurred_mask==False] = (0, 0, 0)
    foreground[blurred_mask==True] = (0, 0, 0)

    p1 = (np.sum(background==0) / background.size)
    p2 = (np.sum(foreground==0) / foreground.size)

    # simple method to tell which image is a brain tissue and which is a skull by number of black pixels
    final_img = background if p1 < p2 else foreground
    new_dataset[i] = final_img
  return new_dataset

def preprocess(dataset):
  '''
    Preprocess a dataset of images as an input to CNN.

    This function normalizes each image in the provided dataset, converts it to grayscale,
    applies Gaussian blurring, and thresholds the image to remove noise. It then identifies 
    the largest contour in the thresholded image and crops the original image to this contour,
    resizing the result to a uniform size.

    Parameters:
      dataset : numpy.ndarray

    Returns:
      numpy.ndarray
    
  '''
  new_dataset = np.copy(dataset)
  for i, image in enumerate(tqdm(dataset)):
  
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    image = normalized.astype(np.uint8)
    # Convert the image to grayscale, and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    # Find the extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    # crop new image out of the original image using the four extreme points (left, right, top, bottom)
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]
    new_image = cv2.resize(new_image, (240, 240), interpolation=cv2.INTER_CUBIC) / 255
    new_dataset[i] = new_image
  return new_dataset