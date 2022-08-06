import cv2
import glob
import numpy as np
from skimage import io

%load_ext autoreload
%autoreload 2
from venomai import preprocess

target_res = 5 # pixels per mm

image_files = np.sort(glob.glob('data/raw/images/*.jpg'))
num_annotators = len(glob.glob('data/raw/masks/*'))

images = []
masks = []

for i in range(len(image_files)):
    
    print(f'{i}/{len(image_files)}')
    
    # Load original image
    image = io.imread(image_files[i])
    
    # Convert to linear RGB color space
    image = preprocess.srgb_to_linear(image)
    
    # Apply automatic white balancing
    image = preprocess.auto_white_balance(image)
    
    # Apply template white balancing
    print('Template white balancing')
    inner_area=10**2
    black_square = preprocess.find_black_square(image[int(image.shape[0]/2):,:,:])
    white_point, black_point, pixel_resolution = preprocess.compute_square_info([black_square], inner_area=inner_area)
    
    image = preprocess.white_balance(image, white_point, black_point)
    
    # Convert back to gamma RGB color space
    image = preprocess.linear_to_srgb(image)
    
    # Rescale image to have a resolution of 6 pixels per mm
    image = preprocess.rescale_image(image, pixel_resolution, target_res=target_res, interpolation=cv2.INTER_CUBIC)
    
    # Save preprocessed image and mask
    io.imsave(f'data/preprocessed/images/{i:04d}.png', image)
    
    for j in range(num_annotators):
    
        mask = io.imread(f'data/raw/masks/{j}/{i:04d}.png')
        mask = preprocess.rescale_image(mask, pixel_resolution, target_res=target_res, interpolation=cv2.INTER_NEAREST)
        mask = (np.round(mask / 255) * 255).astype('uint8')

        io.imsave(f'data/preprocessed/masks/{j}/{i:04d}.png', mask)