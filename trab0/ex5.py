import numpy as np
import cv2

# Loads the image in grayscale
def load_image_grayscale(img_path) :
    img_path = img_path
    return cv2.imread(img_path, 0)

def display_image(img, img_title):
    # Converts to unsigned 8 bit int
    abs_img = cv2.convertScaleAbs(img)
    cv2.imshow(img_title, abs_img)

def end_program():
    # Cleans images
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img_a = load_image_grayscale("img/peppers.png")
img_b = load_image_grayscale("img/house.png")

# First, let us print the images without any kind of modification
display_image(img_a, 'Grayscale Image A')
display_image(img_b, 'Grayscale Image B')

# We must combine those images with different weights for our mean
mean_weights = [(0.2, 0.8), (0.5, 0.5), (0.8, 0.2)]

for weights in mean_weights:
    combined_image = (img_a * weights[0]) + (img_b * weights[1])
    display_image(combined_image, f'{weights[0]}*A + {weights[1]}*B')

end_program()


