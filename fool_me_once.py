from imageio import imread
from skimage import transform
from skimage import exposure
from skimage.color import rgb2gray
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
import numpy as np
from keras import backend as K
from PIL import Image

import logging
logging.basicConfig(level=logging.DEBUG)

def standardise_img(image):
    image = rgb2gray(image)
    image = transform.resize(image, (64, 64))
    image = exposure.equalize_hist(image)
    return image

logging.info("Loading model...")
model = load_model("speed_signs.h5")

# classes in the model
classes = {
    0: "Speed limit (20)",
    1: "Speed limit (30)",
    2: "Speed limit (50)",
    3: "Speed limit (60)",
    4: "Speed limit (70)",
    5: "Speed limit (80)",
    6: "End",
    7: "Speed limit (100)",
    8: "Speed limit (120)",
}

target_idx = ???

def hash_dist(h1, h2):
    s1 = str(h1)
    s2 = str(h2)
    return sum(map(lambda x: 0 if x[0] == x[1] else 1, zip(s1, s2)))

def print_prediction(prediction):
    confidences = list(enumerate(prediction[0]))
    confidences = sorted(confidences, key=lambda x: x[1], reverse=True)

    for class_id, match in confidences[:5]:
        if match > 0.01:
            print(str(round(match,2)) + ": " + classes[class_id] + "(" + str(class_id) + ")")

    return confidences[0]

def create_img(original_image, target_idx):

    #
    # TODO: Implement this function!
    # You probably want some prep here...
    #

    modified_image = None

    while True:
        #
        # And some code here to keep adjusting the modified image a bit at a time,
        # until it reaches a certain level of confidence
        #

        # Print prediction
        print("="*20)
        print("Iteration " + str(iteration))
        prediction = model.predict(modified_image)
        best_match = print_prediction(prediction)

        # See if we're done
        if best_match[0] > 0.95 and best_match[1] == target_idx:
            break

    modified_image = modified_image.reshape((64,64,1))
    img = array_to_img(modified_image)
    img.save("solution.png")

# Get original image
img = Image.open("static/sign.jpg")
img = np.asarray(img)
img = standardise_img(img)
img = np.expand_dims(img, axis=2)

preds = model.predict(np.expand_dims(img,0))
print("Original image")
print_prediction(preds)

create_img(img, target_idx)