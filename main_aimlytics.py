import numpy as np
import cv2

def acquisition(input_image: str):
    img = cv2.imread(input_image)
    if img is None:
        print('Could not open or find the image:', input_image)
        exit(0)

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

if __name__ == '__main__':
    image = acquisition("target_images/clean_target.jpg")  #insert name_file to analyze

    output_image = preprocess_image(image)
    detect_target_contours(image)

    #shows a unique window with input and analyzed image
    concat_images = np.concatenate((image, output_image), axis=1)
    cv2.imshow('Input vs Analyzed Target', concat_images)

    #press a random keyboard key to end the running main
    cv2.waitKey(0)
    cv2.destroyAllWindows()
