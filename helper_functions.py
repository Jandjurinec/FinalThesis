import os
import kagglehub
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.utils import shuffle
import imutils

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

IMG_SIZE = (240, 240)

def plot_samples(X, y, n=30):
  '''
    Creates a gridplot for desired number of images (n) from the specified set
  '''
  for index in [0,1]:
    imgs = X[np.argwhere(y == index)][:n]
    j = 10
    i = int(n/j)

    plt.figure(figsize=(15,6))
    c = 1
    for img in imgs:
      plt.subplot(i,j,c)
      plt.imshow(img[0])

      plt.xticks([])
      plt.yticks([])
      c += 1
    plt.suptitle('Tumor: {}'.format(index))
    plt.show()

def crop_brain_contour(image):
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

  return new_image

def perform_preprocesing(img, img_size, preproces=True):
  if preproces:
    img = crop_brain_contour(img)

  img = cv2.resize(img, img_size, interpolation=cv2.INTER_CUBIC)
  return img / 255

def load_data(img_size=IMG_SIZE, preproces=True, full_size=False):

  # load images from brain-mri-images-for-brain-tumor-detection dataset
  path = kagglehub.dataset_download("navoneel/brain-mri-images-for-brain-tumor-detection")
  path = os.path.join(path, "brain_tumor_dataset")

  X = []
  y = []

  images2remove = [
    "Y248.JPG", # not a brain
    "N15.jpg", # looks like tumor, but in no folder
    "N16.jpg", # looks like tumor, but in no folder
    "N17.jpg", # looks like tumor, but in no folder
    "N11.jpg", # looks like tumor, but in no folder
    "Y185.jpg", # looks like no tumor, but in yes folder
    "Y187.jpg", # looks like no tumor, but in yes folder
    "Y188.jpg", # looks like no tumor, but in yes folder
  ]
  for dir in os.listdir(path):
    for filename in os.listdir(path + '/' + dir):
      
      if filename in images2remove:
        continue
      full_path = os.path.join(path, dir, filename)
      image = cv2.imread(full_path)
      image = perform_preprocesing(image, img_size)
      X.append(image)
      y.append([1]) if dir == 'yes' else y.append([0])

  # load images from brain-tumor-classification-mri dataset
  path = kagglehub.dataset_download("sartajbhuvaji/brain-tumor-classification-mri")
  tumor_subdirs = ['pituitary_tumor', 'meningioma_tumor', 'glioma_tumor']

  for dir in os.listdir(path):
    if dir == 'Training' and not full_size:
      continue # can add more samples from Training
    for subdir in os.listdir(path + '/' + dir):
      for filename in os.listdir(path + '/' + dir + '/' + subdir):
        full_path = os.path.join(path, dir, subdir, filename)
        image = cv2.imread(full_path)
        image = perform_preprocesing(image, IMG_SIZE)
        X.append(image)
        y.append([1]) if subdir in tumor_subdirs else y.append([0])

  X, y = np.array(X), np.array(y)
  X, y = shuffle(X, y)
  print("Data is loaded.")
  return X, y

def plot_metrics(model):
  history = model.history.history
  train_loss = history['loss']
  val_loss = history['val_loss']
  train_acc = history['accuracy']
  val_acc = history['val_accuracy']

  # Loss
  plt.figure()
  plt.plot(train_loss, label='Training Loss')
  plt.plot(val_loss, label='Validation Loss')
  plt.title('Loss')
  plt.legend()
  plt.show()

  # Accuracy
  plt.figure()
  plt.plot(train_acc, label='Training Accuracy')
  plt.plot(val_acc, label='Validation Accuracy')
  plt.title('Accuracy')
  plt.legend()
  plt.show()

def plot_confusion_matrix(y, pred):
  confusion_mtx = confusion_matrix(y, pred)
  cm = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_mtx, display_labels = [0, 1])
  cm.plot()
  plt.show()

def test_model(model, X_test, y_test, verbose):
  predictions = model.predict(X_test, verbose=1 if verbose else 0)
  predictions = [1 if x>0.5 else 0 for x in predictions]

  acc = accuracy_score(y_test, predictions)
  f1 = f1_score(y_test, predictions)
  if verbose:
    print('accuracy = %.2f' % acc)
    print('f1 score = %.2f' % f1)
    plot_confusion_matrix(y_test, predictions)
  return acc, f1

def train(model, X_train, y_train, X_test, y_test, epochs=5, batch_size=32, callbacks=[], verbose=False, validation_split = 0.1):
  if verbose:
    print(f'Data shapes: {X_train.shape}, {y_train.shape}')

    plot_samples(X_train, y_train)

    # Get model info
    print('Model info:')
    model.summary()
    print(f'\nFitting the model...')
  # create test and train data
  model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split = validation_split, callbacks=callbacks, verbose=1 if verbose else 0)

  if verbose:
    # plot the learning data
    plot_metrics(model)

  # test model on test data
  acc, f1 = test_model(model, X_test, y_test, verbose)

  return acc, f1