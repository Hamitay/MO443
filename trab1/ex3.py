import numpy as np
import math
import cv2
from matplotlib import pyplot as plt

MAX_LEVEL = 255

# Loads the image in grayscale
def load_image_grayscale() :
    img_path = "img/peppers.png"
    return cv2.imread(img_path, 0)

# Plots both the image and it's histogram
def add_image_to_plot(img, img_title):
    fig = plt.figure(img_title)

    fig.add_subplot(221)
    plt.title('Image')
    plt.set_cmap('gray')
    plt.imshow(img)

    fig.add_subplot(222)
    plt.title('Histogram')
    plt.hist(img.ravel(), 256, [0,256])


def end_program():
    # Cleans images
    cv2.waitKey(0)
    cv2.destroyAllWindows()

'''
    Executes the operation g = c log(f+1)
'''
def log_transform(img):
    max_value = img.max()

    c = MAX_LEVEL/(math.log10(1+max_value))

    return (c*np.log10(img+1))

'''
    Executes the operation g = c e^f
'''
def exp_transform(img):
    # Changes the type of the array to prevent overflow
    img = img.astype(np.float64)

    img = img/MAX_LEVEL
    img = np.expm1(img)

    c = 255/(math.exp(1)-1)

    return c*img

'''
    Executes the operation g = c f^2
'''
def square_transform(img):
    c = 1/MAX_LEVEL

    img = img.astype(np.float64)
    img = img*img

    return (c*img)

'''
    Executes the operation g = c sqrt(f)
'''
def sqrt_transform(img):
    c = MAX_LEVEL/(math.sqrt(MAX_LEVEL))
    return (c*np.sqrt(img)).astype(int)

def enlarge_contrast(img):
    ximg = img.copy()

    # Arbitrary values chosen based on the original histogram
    a = 74
    alpha = 0.5

    b = 163
    beta = 1.2

    gamma = 0.7

    def contrast_func(pixel):
        if (pixel >= 0 and pixel <= a):
            return alpha*pixel

        if (pixel > a and pixel <= b):
            return beta*(pixel - a) + alpha*a

        if (pixel > b and pixel <= MAX_LEVEL):
            return gamma*(pixel - b) + beta*(b - a) + alpha*a

        return pixel

    vfunc = np.vectorize(contrast_func)
    return vfunc(img)


img = load_image_grayscale()
height, width = img.shape

# First, let us print the image without any kind of modification
add_image_to_plot(img, f'Original Image')


log_image = log_transform(img)
exp_image = exp_transform(img)
square_transform = square_transform(img)
sqrt_transform = sqrt_transform(img)
en_constrats = enlarge_contrast(img)

add_image_to_plot(log_image, f'Log Image')
add_image_to_plot(exp_image, f'Exponential Image')
add_image_to_plot(square_transform, 'Squared image')
add_image_to_plot(sqrt_transform, 'Square Root Image')
add_image_to_plot(en_constrats, 'Enlarged contrast Image')


plt.show()

end_program()
