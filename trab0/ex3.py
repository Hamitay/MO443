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

# We want to get the bit plane slicing of the image
bit_map = 0b00000001

for i in range(8):
    # We will shift our bit map every iteration to get the ith bit plane slicing of the image
    shifted_bitmap = bit_map << i

    # A AND operator with our bit_map will give us the slice we want
    transformed_image = img & shifted_bitmap

    # We convert anything that is not zero to 255 (black)
    transformed_image[transformed_image > 0] = 255
    display_image(transformed_image, f'{i}th bit plane')

end_program()


