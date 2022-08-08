import cv2
import glob
import numpy as np
from skimage import io
import albumentations as A
from scipy.ndimage import map_coordinates

from skimage.util.shape import view_as_blocks

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from torch.utils.data import Dataset, DataLoader

from venomai import utils

def load_preprocessed_data():
    """
    A function that loads preprocessed data for training.
    """
    
    image_files = np.sort(glob.glob('data/preprocessed/images/*.png'))
    num_annotators = len(glob.glob('data/preprocessed/masks/*'))

    images = []
    masks = []

    for i in range(len(image_files)):
        
        image = io.imread(image_files[i])
        images.append(image)
        
        annotator_masks = []
        for j in range(num_annotators):
        
            mask = io.imread(f'data/preprocessed/masks/{j}/{i:04d}.png')
            annotator_masks.append(mask)

        masks.append(np.moveaxis(annotator_masks, 0, -1))
    
    # Replace bad pixels with color most frequent among neighbors
    for i in range(len(masks)):
        for j in range(num_annotators):
            m = masks[i][:,:,:,j]
            mp = np.array([255-255*(np.sum(m,axis=-1)>0), m[...,2], m[...,0]]) / 255
            if np.sum(mp) - mp.shape[1] * mp.shape[2] != 0:
                bad_pixels = np.argwhere(np.sum(m,axis=2) == 510)
                for k in range(len(bad_pixels)):
                    window = m[bad_pixels[k,0]-3:bad_pixels[k,0]+4,bad_pixels[k,1]-3:bad_pixels[k,1]+4]
                    if np.sum(window[...,0]) > np.sum(window[...,2]):
                        m[bad_pixels[k,0],bad_pixels[k,1]] = [255,0,0]
                    else:
                        m[bad_pixels[k,0],bad_pixels[k,1]] = [0,0,255]
                masks[i][:,:,:,j] = m
        
    return images, masks

def extract_patches(image, mask, patch_size=256, shifted=False):
    
    # Padded shape
    padded_shape = patch_size * np.ceil(np.array(image.shape) / patch_size).astype(int)
    if shifted:
        padded_shape[0] += patch_size
        padded_shape[1] += patch_size
    
    # Get number of samples expected
    num_samples = int(np.prod(padded_shape[:2] / patch_size))
    
    # Extract image patches
    padded_image = utils.pad_to_input_shape(image, input_shape=padded_shape, channel_axis=2, mode='reflect')
    image_patches = view_as_blocks(padded_image, block_shape=(patch_size, patch_size, image.shape[2]))
    image_patches = image_patches.reshape(num_samples, patch_size, patch_size, image.shape[2])
    
    # Padded shape
    padded_shape = patch_size * np.ceil(np.array(mask.shape) / patch_size).astype(int)
    if shifted:
        padded_shape[0] += patch_size
        padded_shape[1] += patch_size
        
    # Extract mask patches
    padded_mask = utils.pad_to_input_shape(mask, input_shape=padded_shape, channel_axis=[2,3], mode='reflect')
    mask_patches = view_as_blocks(padded_mask, block_shape=(patch_size, patch_size, mask.shape[2], mask.shape[3]))
    mask_patches = mask_patches.reshape(num_samples, patch_size, patch_size, mask.shape[2], mask.shape[3])

    return image_patches, mask_patches

def get_split(images, masks, split_index, split_ratios=None, num_folds=5, outer_seed=123, inner_seed=321):
    
    if split_ratios is None:
        split_ratios = [0.6,0.2,0.2]
        
    train_ratio, val_ratio, test_ratio = split_ratios
    
    outer_split = train_test_split(images,
                                   masks,
                                   test_size=test_ratio,
                                   shuffle=True,
                                   random_state=outer_seed)
    
    train_images, test_images, train_masks, test_masks = outer_split
    
    kf = KFold(n_splits=5, shuffle=True, random_state=inner_seed)
    
    i = 0
    for train_idx, val_idx in kf.split(train_images):
        if split_index == i:
            break
        i += 1
    
    val_images = [train_images[i] for i in val_idx]
    val_masks = [train_masks[i] for i in val_idx]
    
    train_images = [train_images[i] for i in train_idx]
    train_masks = [train_masks[i] for i in train_idx]
    
    print(len(train_images), len(val_images), len(test_images))
    
    return train_images, train_masks, val_images, val_masks, test_images, test_masks

def setup_loaders(images,
                  masks,
                  train_images,
                  train_masks,
                  val_images,
                  val_masks,
                  test_images,
                  test_masks,
                  input_size=256,
                  batch_size=16):
    
    full_dataset = UNetLoader(images,
                              masks,            
                              input_size=input_size,
                              training=True,
                              augment=True)
    
    train_dataset = UNetLoader(train_images,
                               train_masks,            
                               input_size=input_size,
                               training=True,
                               augment=True)

    val_dataset = UNetLoader(val_images,
                             val_masks,
                             input_size=input_size,
                             training=False,
                             augment=False)
    
    test_dataset = UNetLoader(test_images,
                              test_masks,
                              input_size=input_size,
                              training=False,
                              augment=False)
    
    full_loader = DataLoader(train_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=0)
    
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0)

    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=0)
    
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=0)
    
    return full_loader, train_loader, val_loader, test_loader

def create_data_loaders(split_index, batch_size=8, input_size=256, outer_seed=123, inner_seed=123):
    
    images, masks = load_preprocessed_data()
    train_images, train_masks, val_images, val_masks, test_images, test_masks = get_split(images.copy(),
                                                                                          masks.copy(),
                                                                                          split_index,
                                                                                          outer_seed=outer_seed,
                                                                                          inner_seed=inner_seed)
    full_loader, train_loader, val_loader, test_loader = setup_loaders(images,
                                                                       masks,
                                                                       train_images,
                                                                       train_masks,
                                                                       val_images,
                                                                       val_masks,
                                                                       test_images,
                                                                       test_masks,
                                                                       input_size=input_size,
                                                                       batch_size=batch_size) 

    return full_loader, train_loader, val_loader, test_loader

def augment_slice(image_patch, mask_patch, weight_patch):
    transform = A.Compose([
        A.Flip(p=0.5),
        A.Transpose(p=0.5),
        A.RandomRotate90(p=1),
        A.Rotate(p=0.5, limit=180, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.HueSaturationValue(p=0.25, hue_shift_limit=100),
        A.RandomBrightnessContrast(p=0.25, brightness_limit=0.4, contrast_limit=0.2),
        A.OneOf([
            A.GaussNoise(),
        ], p=0.2),
        A.OneOf([
            A.MotionBlur(),
            A.MedianBlur(blur_limit=3),
            A.Blur(blur_limit=3),
        ], p=0.2),
        A.OneOf([
            A.OpticalDistortion(),
            A.GridDistortion(),
            A.PiecewiseAffine(),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.Sharpen(),
            A.Emboss(),           
        ], p=0.2),
    ])
    
    transformed = transform(image=image_patch, masks=[mask_patch, weight_patch])
    image_patch = transformed['image']
    mask_patch, weight_patch = transformed['masks']
    return image_patch, mask_patch, weight_patch

class UNetLoader(Dataset):

    def __init__(self,
                 images,
                 masks,
                 input_size=256,
                 training=False,
                 augment=False):
        
        self.input_size = input_size
        self.training = training
        self.augment = augment
        
        self.annotator_idx = None
        
        self.image_patches = []
        self.mask_patches = []
        for i in range(len(images)):
            im, ma = extract_patches(images[i], masks[i], patch_size=256, shifted=False)
            self.image_patches += list(im)
            self.mask_patches += list(ma)
            im, ma = extract_patches(images[i], masks[i], patch_size=256, shifted=True)
            self.image_patches += list(im)
            self.mask_patches += list(ma)
            
        self.image_patches = np.array(self.image_patches, dtype='uint8')
        self.mask_patches = np.array(self.mask_patches, dtype='uint8')
        
        self.labels_1 = np.argwhere(np.sum(self.mask_patches[:,:,:,0,:], axis=(1,2,3)) > 0).ravel()
        self.labels_2 = np.argwhere(np.sum(self.mask_patches[:,:,:,2,:], axis=(1,2,3)) > 0).ravel()
        
    def evaluate(self, annotator_idx=None):
#         self.training = False
        
        if annotator_idx in list(np.arange(self.mask_patches[0].shape[-1])):
            self.annotator_idx = annotator_idx
        else:
            print('Given annotator index doesn\'t exist.')
            raise ValueError
        
#     def train(self):
#         self.training = True
#         self.annotator_idx = None
        
    def augmentation(self, augment):
        self.augment = augment
    
    def __len__(self):
        return int(self.image_patches.shape[0] / 8)
    
    def __getitem__(self, idx):
        
        # Select random index
        if self.training:
            if np.random.rand() < 0.75:
                if np.random.rand() < 0.5:
                    idx = np.random.choice(self.labels_1)
                else:
                    idx = np.random.choice(self.labels_2)
            else:
                idx = np.random.randint(len(self.image_patches))
        
        # Select annotator
        if self.annotator_idx is None:
            annotator_idx = np.random.randint(self.mask_patches[idx].shape[-1])
        else:
            annotator_idx = self.annotator_idx
           
        # Get image, mask and weight
        image_patch = self.image_patches[idx]
        mask_patch = self.mask_patches[idx,:,:,:,annotator_idx]
        weight_patch =  255 * np.ones((mask_patch.shape[0], mask_patch.shape[1]), dtype='uint8')
        
        # Augment patch
        if self.augment:
            image_patch, mask_patch, weight_patch = augment_slice(image_patch, mask_patch, weight_patch)
        
        # Make image into channels, shape format
        image_patch = np.moveaxis(image_patch, -1, 0)
        
        # Make mask into 3 class for softmax 
        mask_patch = np.array([255-255*(np.sum(mask_patch,axis=-1)>0), mask_patch[...,2], mask_patch[...,0]])
        
        # Make weight into 3 class for softmax
        weight_patch = np.array([weight_patch, weight_patch, weight_patch])

#         import matplotlib.pyplot as plt
#         print(image_patch.shape, np.min(image_patch), np.max(image_patch))
#         plt.imshow(image_patch[0])
#         plt.show()

#         print(np.min(mask_patch[1]), np.max(mask_patch[1]))
#         plt.imshow(mask_patch[1])
#         plt.show()
        
#         print(np.min(mask_patch[2]), np.max(mask_patch[2]))
#         plt.imshow(mask_patch[2])
#         plt.show()

#         print(np.min(weight_patch[0]), np.max(weight_patch[0]))
#         plt.imshow(weight_patch[0])
#         plt.show()
        
        image_patch = image_patch / 255.0
        mask_patch = mask_patch / 255.0
        weight_patch = (weight_patch > 0).astype(np.float32)
        
        sample = (image_patch.astype(np.float32), mask_patch.astype(np.float32), weight_patch.astype(np.float32))

        return sample