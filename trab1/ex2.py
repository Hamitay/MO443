import numpy as np
import cv2

MAX_INTENSITY = 255

# Loads the image in grayscale
def load_image_grayscale() :
    img_path = "img/patrick.jpeg"
    return cv2.imread(img_path, 0)

def display_image(img, img_title):
    # Converts to unsigned 8 bit int
    abs_img = cv2.convertScaleAbs(img)
    cv2.imwrite(f'{img_title}.jpg', abs_img)
    cv2.imshow(img_title, abs_img)

# For visualization purposes we will change the maximum intensity
# of an image to white
def increase_white_contrast(img):
    ximg = img.copy()
    ximg[ximg == ximg.max()] = MAX_INTENSITY
    return ximg

def end_program():
    # Cleans images
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = load_image_grayscale()
height, width = img.shape

# First, let us print the image without any kind of modification
display_image(img, f'Original Image with 256 levels')

# We will use bit masks to remove the LSB of our pixels
masks = [
    (0b11111100, 64),
    (0b11111000, 32),
    (0b11110000, 16),
    (0b11100000, 8),
    (0b11000000, 4),
    (0b10000000, 2)
]

for mask in masks:
    ximg = img & mask[0]

    # We will increase the white contrast to help visualize
    ximg = increase_white_contrast(ximg)

    display_image(ximg, f'Image with {mask[1]} levels')

end_program()
