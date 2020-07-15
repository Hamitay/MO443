import numpy as np
import cv2

# Loads the image in grayscale
def load_image_grayscale() :
    img_path = "img/peppers.png"
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

def mirror_filter(mask):
    return np.rot90(mask, 2)

def apply_filter(img, mask):
    mirrored_filter = mirror_filter(mask)
    return cv2.filter2D(img, -1, mirrored_filter)

# Loading the image
img = load_image_grayscale()

# Filters h1 to h8
h = []
h.append(np.array(([[0, 0, -1, 0, 0], [0, -1, -2, -1, 0], [-1, -2, 16, -2, -1],[0, -1, -2, -1, 0], [0, 0, -1, 0, 0]]), float))
h.append(np.array(([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]]), float)*(1/256))
h.append(np.array(([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]), float))
h.append(np.array(([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]), float))
h.append(np.array(([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]), float))
h.append(np.array(([[1, 1, 1], [1, 1, 1], [1, 1, 1]]), float)*1/9)
h.append(np.array(([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]]), float))
h.append(np.array(([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]]), float))

display_image(img, 'original')
for i in range(len(h)):
    filtered_image = apply_filter(img, h[i])
    display_image(filtered_image, f'h{i+1}')

# Sobel operator
# sqrt(h3^2 + h4^2)
filter_h3 = mirror_filter(h[2])
filter_h4 = mirror_filter(h[3])

# We must normalize the scale in this case
filtered_imageX = cv2.filter2D(img, -1, filter_h3)/255
filtered_imageY = cv2.filter2D(img, -1, filter_h4)/255

# Normalized operation
sobel = (np.sqrt(filtered_imageX**2 + filtered_imageY**2)*255).astype(int)
display_image(sobel, 'Sobel Operator Edge Detection')

end_program()
