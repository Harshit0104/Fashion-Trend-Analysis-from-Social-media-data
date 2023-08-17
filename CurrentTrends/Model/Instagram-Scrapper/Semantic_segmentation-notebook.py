# %% [markdown]
# ## Colab Setup

# %%
%tensorflow_version 1.x

# %%
!pip install Keras==2.1

# %%
!pip list

# %%
# mounting and connecting with drive
from google.colab import drive
drive.mount('/content/drive')

# %%
# # Setup colab
# !pip install -U -q PyDrive
# from pydrive.auth import GoogleAuth
# from pydrive.drive import GoogleDrive
# from google.colab import auth
# from oauth2client.client import GoogleCredentials
# # Authenticate and create the PyDrive client.
# auth.authenticate_user()
# gauth = GoogleAuth()
# gauth.credentials = GoogleCredentials.get_application_default()
# drive = GoogleDrive(gauth)

# %%
import warnings 
warnings.filterwarnings("ignore")

# %%
# importing libraries
import os
import gc
import sys
import math
import json
import glob
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import skimage.io
from IPython.display import clear_output

import itertools
from tqdm import tqdm

from imgaug import augmenters as iaa
from sklearn.model_selection import StratifiedKFold, KFold

import tensorflow as tf

# %%
# setting up path directory
TRAIN_IMAGE_DIR = Path('/content/drive/My Drive/flipkart/Train large/')
ROOT_DIR = Path('/content/')
DATA_DIR = Path('/content/drive/My Drive/flipkart/')

# %% [markdown]
# ## Data Import

# %%
# import train file 
import pandas as pd
train = pd.read_csv('/content/drive/My Drive/flipkart/train.csv')
train.head()

# %%
# extracting image metadata fom json file 
with open(DATA_DIR/"labels.json") as f:
    label_descriptions = json.load(f)

label_names = [x['name'] for x in label_descriptions['categories']]

# %%
label_df = pd.DataFrame(label_names).reset_index()
label_df.columns = ['Id','Labels']
label_df.head()

# %%
segment_df = train
segment_df['CategoryId'] = segment_df['ClassId'].str.split('_').str[0]

print("Total segments: ", len(segment_df))
segment_df.head()

# %%
# Rows with the same image are grouped together because the subsequent operations perform at an image level
image_df = segment_df.groupby('ImageId')['EncodedPixels', 'CategoryId'].agg(lambda x: list(x))
size_df = segment_df.groupby('ImageId')['Height', 'Width'].mean()
image_df = image_df.join(size_df, on='ImageId')

print("Total images: ", len(image_df))
image_df.head()

# %% [markdown]
# ## EDA

# %%
plt.figure(figsize=(15, 7))
sns.jointplot(x=image_df['Width'], y=image_df['Height'])

# %%
plt.figure(figsize=(7, 5))
sns.distplot(image_df['Height'], kde=False);
plt.title("Height Distribution", fontsize=10)
plt.show()

# %%
plt.figure(figsize=(7, 5))
sns.distplot(image_df['Width'], kde=False);
plt.title("Width Distribution", fontsize=10)
plt.show()

# %%
plt.figure(figsize=(10, 5))
sns.distplot((image_df['Height'] * image_df['Width'])/10000, kde=False);
plt.title("Area Distribution /(10000)", fontsize=10)
plt.xlabel(" Area (in 10k)", fontsize=10)
plt.show()

# %%
# number of labels per image
labels_per_image = image_df['CategoryId'].map(lambda x:len(x)).value_counts().to_frame().reset_index().sort_values(by = 'index')
labels_per_image.columns = ['#labels','#images']

plt.figure(figsize=(15, 7))
sns.barplot(labels_per_image['#labels'],labels_per_image['#images'])
plt.title("Number of Labels per Image", fontsize=20)
plt.xlabel("# of labels", fontsize=20)
plt.ylabel("# of images", fontsize=20)
plt.show()

# %%
segment_df['CategoryId'] = segment_df['CategoryId'].astype('int64')
labels_per_image2 = segment_df.merge(label_df, how='left', left_on='CategoryId', right_on='Id')
labels_per_image3 = labels_per_image2.groupby('Labels')['ImageId'].nunique().to_frame().reset_index()
labels_per_image3.head()

# %%
labels_per_image4 = labels_per_image2.groupby('Labels')['ImageId'].count().to_frame().reset_index()
labels_per_image4.head()

# %%
labels_per_image4.to_csv('word_cloud_data.csv')

# %%
d = {}
for i in range(len(labels_per_image4)):
    d[labels_per_image4.iloc[i,0]] = labels_per_image4.iloc[i,1]

# %%
from wordcloud import WordCloud

wordcloud = WordCloud(background_color='Ghostwhite')
wordcloud.generate_from_frequencies(frequencies=d)

plt.figure(figsize=(25, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# %%
plt.figure(figsize=(20, 7))
sns.barplot(labels_per_image3['Labels'],labels_per_image3['ImageId'])
plt.xticks(rotation=90)
plt.title("Labels Distribution in Images", fontsize=20)
plt.xlabel("labels", fontsize=10)
plt.ylabel("# of images", fontsize=10)
plt.show()

# %% [markdown]
# ## Data Setup

# %%
# Since we are training on ~5k images, we will fetch train data for those 5k images

images = os.listdir(TRAIN_IMAGE_DIR)
uploaded_images = pd.DataFrame(images, columns = ['image_name'])
image_df = image_df[image_df.index.isin(uploaded_images['image_name'])]

# %%
image_df.shape

# %%
# Partition data in train and test
FOLD = 0
N_FOLDS = 5

kf = KFold(n_splits=N_FOLDS, random_state=42, shuffle=True)
splits = kf.split(image_df) # ideally, this should be multilabel stratification

def get_fold():    
    for i, (train_index, valid_index) in enumerate(splits):
        if i == FOLD:
            return image_df.iloc[train_index], image_df.iloc[valid_index]
        
train_df, valid_df = get_fold()

# %% [markdown]
# ## Setting up Mask RCNN

# %%
# import matterport Mask-RCNN implementation
!git clone https://www.github.com/matterport/Mask_RCNN.git;
os.chdir('Mask_RCNN')

!rm -rf .git # to prevent an error when the kernel is committed
!rm -rf images assets # to prevent displaying images at the bottom of a kernel

sys.path.append(ROOT_DIR/'Mask_RCNN')
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

# %%
!pwd

# %%
os.chdir(ROOT_DIR)
!wget --quiet https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
!ls -lh mask_rcnn_coco.h5

COCO_WEIGHTS_PATH = '/content/mask_rcnn_coco.h5'

# %%
# # Already have trained weights, we will continue on those weights
# pre_trained_weight = '/content/drive/My Drive/Projects/iMaterialist/trained weights/weights_0.08133.h5'

# %%
# Set configuration

NUM_CATS = 46  # classification ignoring attributes (only categories)
IMAGE_SIZE = 512 # the image size is set to 512, which is the same as the size of submission masks

class FashionConfig(Config):
    NAME = "fashion"
    NUM_CLASSES = NUM_CATS + 1 # +1 for the background class
    
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4 # Batch size - memory error occurs when IMAGES_PER_GPU is too high
    #https://datascience.stackexchange.com/questions/29719/how-to-set-batch-size-steps-per-epoch-and-validation-steps
    
    BACKBONE = 'resnet50' #resnet50 will be lighter than resnet101 for training
    
    IMAGE_MIN_DIM = IMAGE_SIZE
    IMAGE_MAX_DIM = IMAGE_SIZE    
    IMAGE_RESIZE_MODE = "none"
    
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    #DETECTION_NMS_THRESHOLD = 0.7

    STEPS_PER_EPOCH = 500
    VALIDATION_STEPS = 100

    MAX_GT_INSTANCES = 50
    DETECTION_MAX_INSTANCES = 50

    ## balance out losses
    # https://stackoverflow.com/questions/55360262/what-exactly-are-the-losses-in-matterport-mask-r-cnn
    # https://stackoverflow.com/questions/46272841/what-is-the-loss-function-of-the-mask-rcnn
    LOSS_WEIGHTS = {
          "rpn_class_loss": 1.0, # How well the Region Proposal Network separates background with objetcs
          "rpn_bbox_loss": 0.8, # How well the RPN localize objects
          "mrcnn_class_loss": 6.0, # How well the Mask RCNN localize objects
          "mrcnn_bbox_loss": 6.0, # How well the Mask RCNN recognize each class of object
          "mrcnn_mask_loss": 6.0 # How well the Mask RCNN segment objects
    }
    
config = FashionConfig()
config.display()

# %%
# resizing image to 512X512;
def resize_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)  
    return img

# %%
#  MaskRCNN Class

class FashionDataset(utils.Dataset):

    def __init__(self, df):
        super().__init__(self)
        
        # Add classes
        for i, name in enumerate(label_names):
            self.add_class("fashion", i+1, name)
        
        # Add images 
        for i, row in df.iterrows():
            self.add_image("fashion", 
                           image_id=row.name, 
                           path=str(TRAIN_IMAGE_DIR/row.name), 
                           labels=row['CategoryId'],
                           annotations=row['EncodedPixels'], 
                           height=row['Height'], width=row['Width'])

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path'], [label_names[int(x)] for x in info['labels']]
    
    def load_image(self, image_id):
        return resize_image(self.image_info[image_id]['path'])

    def load_mask(self, image_id):
        info = self.image_info[image_id]
                
        mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE, len(info['annotations'])), dtype=np.uint8)
        labels = []
        
        for m, (annotation, label) in enumerate(zip(info['annotations'], info['labels'])):
            sub_mask = np.full(info['height']*info['width'], 0, dtype=np.uint8)
            annotation = [int(x) for x in annotation.split(' ')]
            
            for i, start_pixel in enumerate(annotation[::2]):
                sub_mask[start_pixel: start_pixel+annotation[2*i+1]] = 1

            sub_mask = sub_mask.reshape((info['height'], info['width']), order='F')
            sub_mask = cv2.resize(sub_mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
            
            mask[:, :, m] = sub_mask
            labels.append(int(label)+1)
            
        return mask, np.array(labels)

# %%
# Visualizing random images
dataset = FashionDataset(image_df)
dataset.prepare()

for i in range(1):
    image_id = random.choice(dataset.image_ids)
    print(dataset.image_reference(image_id))
    
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset.class_names, limit=4)

# %%
# Prepare Data
train_dataset = FashionDataset(train_df)
train_dataset.prepare()

valid_dataset = FashionDataset(valid_df)
valid_dataset.prepare()

# %% [markdown]
# ## Training Model

# %%
# Image augmentation
augmentation = iaa.Sequential([
    iaa.OneOf([ ## rotate
        iaa.Affine(rotate=0),
        iaa.Affine(rotate=90),
        iaa.Affine(rotate=180),
        iaa.Affine(rotate=270),
    ]),
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.OneOf([ ## brightness or contrast
        iaa.Multiply((0.9, 1.1)),
        iaa.ContrastNormalization((0.9, 1.1)),
    ]),
    iaa.OneOf([ ## blur or sharpen
        iaa.GaussianBlur(sigma=(0.0, 0.3)),
        iaa.Sharpen(alpha=(0.0, 0.3)),
    ]),
])

# %%
# sample augmentation output
imggrid = augmentation.draw_grid(image, cols=5, rows=2)
plt.figure(figsize=(20, 10))
_ = plt.imshow(imggrid.astype(int))

# %%
# initiating Mask R-CNN training

model = modellib.MaskRCNN(mode='training', config=config, model_dir=ROOT_DIR);
model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=[
    'mrcnn_class_logits', 'mrcnn_bbox_fc', 'mrcnn_bbox', 'mrcnn_mask'])

# %%
# Declaring learning rate
LR = 0.0001

# %%
## train head layer alone

# %%time
# model.train(train_dataset, valid_dataset,
#             learning_rate=LR*2,
#             epochs=2, # EPOCHS[0],
#             layers='heads',
#             augmentation=augmentation)
# history = model.keras_model.history.history
# history

# %%
# Train all layers
%%time
model.train(train_dataset, valid_dataset,
            learning_rate=LR/4,
            epochs=2,
            layers='all',
            augmentation=augmentation)

# new_history = model.keras_model.history.history
# for k in new_history: history[k] = history[k] + new_history[k]
history = model.keras_model.history.history

# %%
#reducing learning rate and training again

%%time
model.train(train_dataset, valid_dataset,
            learning_rate=LR/8,
            epochs=5,
            layers='all',
            augmentation=augmentation)

new_history = model.keras_model.history.history
for k in new_history: history[k] = history[k] + new_history[k]

# %%
epochs = range(1, len(history['loss'])+1)
pd.DataFrame(history, index=epochs)

# %%
# find best epoch
best_epoch = np.argmin(history["val_loss"]) + 1
print("Best epoch: ", best_epoch)
print("Valid loss: ", history["val_loss"][best_epoch-1])

# %%
# Picking the location of our best model weights
glob_list = glob.glob(f'/content/fashion*/mask_rcnn_fashion_{best_epoch:04d}.h5')
model_path = glob_list[0] if glob_list else ''

# %%
# model_path = '/content/fashion20191109T2055/mask_rcnn_fashion_0007.h5'

# %% [markdown]
# ## Prediction

# %%
# Prediction, this cell defines InferenceConfig and loads the best trained model.

class InferenceConfig(FashionConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

model = modellib.MaskRCNN(mode='inference', 
                          config=inference_config,
                          model_dir=ROOT_DIR)

model.load_weights(model_path, by_name=True)

# %%
# Since the submission system does not permit overlapped masks, we have to fix them
def refine_masks(masks, rois):
    areas = np.sum(masks.reshape(-1, masks.shape[-1]), axis=0)
    mask_index = np.argsort(areas)
    union_mask = np.zeros(masks.shape[:-1], dtype=bool)
    for m in mask_index:
        masks[:, :, m] = np.logical_and(masks[:, :, m], np.logical_not(union_mask))
        union_mask = np.logical_or(masks[:, :, m], union_mask)
    for m in range(masks.shape[-1]):
        mask_pos = np.where(masks[:, :, m]==True)
        if np.any(mask_pos):
            y1, x1 = np.min(mask_pos, axis=1)
            y2, x2 = np.max(mask_pos, axis=1)
            rois[m, :] = [y1, x1, y2, x2]
    return masks, rois

# %%
# Let’s load an image and try to see how the model performs. You can use any of your images to test the model.

# Load a random image from the images folder
import skimage.io
image_path = str(ROOT_DIR/'test_image.jpg')

# original image
plt.figure(figsize=(12,10))
skimage.io.imshow(skimage.io.imread(image_path))

img = skimage.io.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

result = model.detect([resize_image(image_path)])
r = result[0]

if r['masks'].size > 0:
    masks = np.zeros((img.shape[0], img.shape[1], r['masks'].shape[-1]), dtype=np.uint8)
    for m in range(r['masks'].shape[-1]):
        masks[:, :, m] = cv2.resize(r['masks'][:, :, m].astype('uint8'), 
                                    (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    y_scale = img.shape[0]/IMAGE_SIZE
    x_scale = img.shape[1]/IMAGE_SIZE
    rois = (r['rois'] * [y_scale, x_scale, y_scale, x_scale]).astype(int)
    
    masks, rois = refine_masks(masks, rois)
else:
    masks, rois = r['masks'], r['rois']
    
visualize.display_instances(img, rois, masks, r['class_ids'], 
                            ['bg']+label_names, r['scores'])


