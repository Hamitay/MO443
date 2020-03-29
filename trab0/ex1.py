import numpy as np
import cv2

# Loads the image in grayscale
def load_image_grayscale() :
    img_path = "img/peppers.png"
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
height, width = img.shape

# First, let us print the image without any kind of modification
display_image(img, 'Grayscale Image')

# Now we shall obtain the image's negative (i)
neg_img = 255-img
display_image(neg_img, 'Negative Image')

# Now we will convert the range of pixels to [100, 200] (ii)
new_range_image = (img*(100/255) + 100)
display_image(new_range_image, 'New Range Image')

# Now we will flip on the even rows of the image (iii)
flipped_image = np.copy(img)
for j in range(height):
    if (j%2 == 0):
        flipped_image[j] = np.flipud(img[j])
display_image(flipped_image, 'Flipped Even Rows Image')

# Lastly we will mirror the rows on the y axis(iii)

# First we get the upper half of the img
upper_half = np.split(img, 2)[0]

# Then we flip it along the y axis
flipped_upper_half = np.flipud(upper_half)

# Then we concatenate the array on the y axis (iv)
mirrored_image = np.append(upper_half, flipped_upper_half, axis=0)
display_image(mirrored_image, 'Mirrored Image')

end_program()


