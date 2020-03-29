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

# First, let us print the image without any kind of modification
display_image(img, 'Grayscale Image')

# Now we must normalize the image pixels to the scale [0, 1] (a)
normalized_image = img/255

# And then we apply the transformation B = A^(1/gamma) for different values of gamma
# With this we can change the image brightness
gamma_values = [1.5, 2.5, 3.5]

for gamma in gamma_values:
    # We apply the transformation then change back to the [0, 255] scale (b)
    transformed_image = (normalized_image**(1/gamma))*255
    display_image(transformed_image, f'Gamma = {gamma}')

end_program()


