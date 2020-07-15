import numpy as np
import cv2
from matplotlib import pyplot as plt

# Loads the image in grayscale
def load_image_grayscale():
    img_path = "img/house.png"
    return cv2.imread(img_path, 0)

def create_circle(img, radius, color):
  # Creates a circle image for our filters masks
  rows, cols = img.shape
  cv2.circle(img, (int(cols/2), int(rows/2)), radius, color, -1)

  return img

def high_pass_mask(img, mask_radius):
  # We can achieve a high pass filter by using a circular mask at the
  # center of the magnitude_spectrum, meaning the highest energy frequencies

  rows, cols = img.shape
  mask = np.ones((rows, cols), np.uint8)
  return create_circle(mask, mask_radius, 0)

def band_pass_mask(img, inner_radius, outer_radius):
  # We can achieve a band pass filter by using two circular mask outside the
  # center of the magnitude_spectrum, meaning the highest energy frequencies

  rows, cols = img.shape
  mask = np.zeros((rows, cols), np.uint8)

  mask = create_circle(mask, outer_radius, 1)
  return create_circle(mask, inner_radius, 0)

def low_pass_mask(img, mask_radius):
  # We can achieve a low pass filter by using a circular mask outside the
  # center of the magnitude_spectrum, meaning the highest energy frequencies

  rows, cols = img.shape
  mask = np.zeros((rows, cols), np.uint8)
  return create_circle(mask, mask_radius, 1)

def inverse_fft(dft_dc):
  inverse_fft = np.fft.ifftshift(dft_dc)
  inverse_fft = np.fft.ifft2(inverse_fft)
  return np.abs(inverse_fft)

def apply_filter_and_inverse_fft(dft_dc, mask):
  # In the frequency dominium the convolution can be calculated by a simple multiplication
  convolution = dft_dc*mask

  return inverse_fft(convolution)

# First, let us load the image in grayscale
img = load_image_grayscale()

# Then let us get the DFT of the image using the FFT package
dft = np.fft.fft2(img)

# We now shift the component of zero frequency (DC component) to the center
dft_dc = np.fft.fftshift(dft)

# Converting the spectrum to decibels
magnitude_spectrum = 20*np.log(np.abs(dft_dc))

# Let us the inverse transform to verify if any data was lost
inverse_img = np.abs(np.fft.ifft2(dft_dc))

# We will now create our filters' masks
high_pass_mask = high_pass_mask(img, 80)
low_pass_mask = low_pass_mask(img, 20)
band_pass_mask = band_pass_mask(img, 25, 90)

# Now we can apply our high pass filter
high_pass_img = apply_filter_and_inverse_fft(dft_dc, high_pass_mask)
low_pass_img = apply_filter_and_inverse_fft(dft_dc, low_pass_mask)
band_pass_img = apply_filter_and_inverse_fft(dft_dc, band_pass_mask)


plt.subplot(331),plt.imshow(img, cmap = 'gray')
plt.title('Imagem Original'), plt.xticks([]), plt.yticks([])

plt.subplot(332),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Espectro de frequência'), plt.xticks([]), plt.yticks([])

plt.subplot(333),plt.imshow(inverse_img, cmap = 'gray')
plt.title('Transformada Inversa'), plt.xticks([]), plt.yticks([])

plt.subplot(334),plt.imshow(high_pass_mask, cmap = 'gray')
plt.title('Nucleo do passa alta'), plt.xticks([]), plt.yticks([])

plt.subplot(335),plt.imshow(low_pass_mask, cmap = 'gray')
plt.title('Nucleo do passa baixa'), plt.xticks([]), plt.yticks([])

plt.subplot(336),plt.imshow(band_pass_mask, cmap = 'gray')
plt.title('Nucleo do passa faixa'), plt.xticks([]), plt.yticks([])

plt.subplot(337),plt.imshow(high_pass_img, cmap = 'gray')
plt.title('Imagem filtrada com passa alta'), plt.xticks([]), plt.yticks([])

plt.subplot(338),plt.imshow(low_pass_img, cmap = 'gray')
plt.title('Imagem filtrada com passa baixa'), plt.xticks([]), plt.yticks([])

plt.subplot(339),plt.imshow(band_pass_img, cmap = 'gray')
plt.title('Imagem filtrada com passa faixa'), plt.xticks([]), plt.yticks([])


plt.show()

