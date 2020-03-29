import numpy as np
import cv2

# Loads the image in grayscale
def load_image_grayscale() :
    img_path = "img/house.png"
    return cv2.imread(img_path, 0)

def display_image(img, img_title):
    # Converts to unsigned 8 bit int
    abs_img = cv2.convertScaleAbs(img)
    cv2.imshow(img_title, abs_img)

def end_program():
    # Cleans images
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = load_image_grayscale()

# First, let us print the image without any kind of modification
display_image(img, 'Grayscale Image')

# We want to create a mosaic of the image by slicing it into 16 blocks in a 4x4 configuration
# We will use a dictionary (hash map) to better find each piece of the mosaic we want on O(1) time
split_images = {}

# First let us split the image in 4 rows
img_rows = np.split(img, 4)

for x in range(4):
    # For each row, let us split in 4 columns
    img_columns = np.split(img_rows[x], 4, axis=1)
    for y in range(4):
        #Then we add the split image into our map
        img_key = 1+(4*x+y)
        split_images[img_key] = img_columns[y]

# With the image split and referenced on our hashmap we will rebuild it on the pattern
# proposed by the exercise
mosaic_indexes_map = [[ 6, 11, 13, 3],
                      [ 8, 16,  1, 9],
                      [12, 14,  2, 7],
                      [ 4, 15, 10, 5]]

# First we build the mosaic rows
mosaic_rows = []

for row in mosaic_indexes_map:
    new_row = split_images[row[0]]
    new_row = np.append(new_row, split_images[row[1]], axis=1)
    new_row = np.append(new_row, split_images[row[2]], axis=1)
    new_row = np.append(new_row, split_images[row[3]], axis=1)

    mosaic_rows.append(new_row)

# Now we append the rows in a single image
mosaic_image = mosaic_rows[0]
mosaic_image = np.append(mosaic_image, mosaic_rows[1], axis=0)
mosaic_image = np.append(mosaic_image, mosaic_rows[2], axis=0)
mosaic_image = np.append(mosaic_image, mosaic_rows[3], axis=0)

display_image(mosaic_image, 'Mosaic')

end_program()


