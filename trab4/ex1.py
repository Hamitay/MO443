import numpy as np
import cv2
from matplotlib import pyplot as plt

# Loads the image in grayscale
def load_image():
    img_path = "img/bitmap.pbm"
    return cv2.imread(img_path, 0)


def display_image(img, img_title):
    # Converts to unsigned 8 bit int
    #abs_img = cv2.convertScaleAbs(img)
    cv2.imwrite(f'{img_title}.jpg', img)
    #cv2.imshow(img_title, abs_img)

def end_program():
    # Cleans images
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def createKernel(height, width):
  return np.ones((height, width))

def dilate(img, kernel):
  return cv2.dilate(img, kernel, iterations=1)

def erode(img, kernel):
  return cv2.erode(img, kernel, iterations=1)

def close(img, kernel):
  return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

def connected_component(img):
  return cv2.connectedComponents(img)

img = load_image()
display_image(img, 'Imagem Original em jpg')

#Invert the image
img = (255 - img)

display_image(img, 'Imagem Negativada')

# Step 1
kernel = createKernel(1, 100)
step_1 = dilate(img, kernel)
display_image(step_1, 'Passo 1 Dilatação - Elemento Estruturante 100x1')

# Step 2
step_2 = erode(step_1, kernel)
display_image(step_2, 'Passo 2 Erosão - Elemento Estruturante 100x1')

# Step 3
kernel = createKernel(200, 1)
step_3 = dilate(img, kernel)
display_image(step_3, 'Passo 3 Dilatação - Elemento Estruturante 1x200')

# Step 4
step_4 = erode(step_3, kernel)
display_image(step_4, 'Passo 4 Erosão - Elemento Estruturante 1x200')

# Step 5
step_5 = step_2 & step_4
display_image(step_5, 'Passo 5 Operação AND - Passos 2 e 4')

# Step 6
kernel = createKernel(1, 30)
step_6 = close(step_5, kernel)
display_image(step_6, 'Passo 6 Fechamento - Passo 5')

# Step 7
ret, labels = connected_component(step_6)

total_size = img.size

text_rectangles = []
word_rectangles = []

display_image(img, 'Componentes Conexos e suas Labels')
for label in range(1,ret):
    mask = np.array(labels, dtype=np.uint8)

    # Make our connected component white
    mask[mask != label] = 0
    mask[labels == label] = 255

    #Get the rectangle and find the component in the original image
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    (x,y,w,h) = cv2.boundingRect(contours[0])
    rectangle = img[y:y+h,x:x+w]

    # Count number of white pixels (black in original image)
    num_white = np.count_nonzero(rectangle == 255)
    ratio = num_white/(w*h)

    # Use empyrical ration to find word lines
    has_word = ratio < 0.52 and ratio > 0.1

    # If it has words we separate them
    if has_word:
      text_rectangles.append((x,y,w,h))

      # Count number of words
      # Dilate and close to aggroup the words
      kernel = createKernel(10, 10)
      step_1 = dilate(rectangle, kernel)
      step_2 = close(step_1, createKernel(2,2))

      word_ret, word_labels = connected_component(step_2)
      for word_label in range(1, word_ret):

        word_mask = np.array(word_labels, dtype=np.uint8)
        word_mask[word_mask != word_label] = 0
        word_mask[word_labels == word_label] = 255

        #Get the rectangle and find the component in the original image
        word_contours = cv2.findContours(word_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
        (wx,wy,ww,wh) = cv2.boundingRect(word_contours[0])
        word_rectangles.append((wx+x, wy+y, ww, wh))

for rectangle in word_rectangles:
  x,y,w,h = rectangle
  img = cv2.rectangle(img, (x,y), (x+w, y+h), (255, 255, 255), 2)

display_image(img, 'Componentes Conexos de Palavras de Texto')

end_program()
