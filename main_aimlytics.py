import numpy as np
import cv2
import matplotlib.pyplot as plt


def acquisition(input_image: str):
    img = cv2.imread(input_image)
    if img is None:
        print("Could not load image")

    return img


def preprocess_image(input_image):
    smoothed_image = cv2.GaussianBlur(input_image, (5, 5), 0)
    no_noise_image = cv2.medianBlur(smoothed_image, 5)
    return no_noise_image


def detect_target_contours(preprocessed_image):
    return


#valutare se mettere funzione
def correct_perspective(input_image):
    return


def assign_points(preprocessed_image):
    return


image = acquisition("target_images/clean_target.jpg")  #insert name_file to analyze
output_image = preprocess_image(image)
detect_target_contours(image)

plt.figure(figsize=(10, 13))
plt.subplot(121)
plt.title("Input Target")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.subplot(122)
plt.title("Analyzed Target")
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY))
plt.axis('off')
