import numpy as np
import cv2

# Loads the image in grayscale
def load_image_grayscale() :
    img_path = "img/patrick.jpeg"
    return cv2.imread(img_path, 0)

def display_image(img, img_title):
    # Converts to unsigned 8 bit int
    abs_img = cv2.convertScaleAbs(img)
    cv2.imwrite(f'{img_title}.jpg', abs_img)
    cv2.imshow(img_title, abs_img)

def end_program():
    # Cleans images
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = load_image_grayscale()
height, width = img.shape

# First, let us print the image without any kind of modification
display_image(img, f'Original Image {height}x{width}')

# Changing the resolution to new size
new_dimensions = [256, 128, 64, 32, 16, 8]

for dimension in new_dimensions:
    # We resize the image to the desired dimensions
    resized_image = cv2.resize(img, (dimension, dimension))

    # We resize the image again to it's original dimension
    # Notice how each time we decrease the resolution we 'lose' information
    display_image(cv2.resize(resized_image, (height, width)), f'Image {dimension}x{dimension}')

end_program()


