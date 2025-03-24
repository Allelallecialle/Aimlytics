import numpy as np
import cv2

def acquisition(input_image: str):
    img = cv2.imread(input_image)
    if img is None:
        print('Could not open or find the image:', input_image)
        exit(0)

    return img


def preprocess_image(input_image):
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    smoothed_image_gauss = cv2.GaussianBlur(gray_image, (5, 5), 0)
    smoothed_image_bi = cv2.bilateralFilter(smoothed_image_gauss, 9, 75, 75)
    binary_image = cv2.adaptiveThreshold(smoothed_image_bi, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY_INV, blockSize=11, C=2)
    kernel = np.ones((3, 3), np.uint8)
    cleaned_binary = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=2)
    return cleaned_binary


def detect_target_contours(cleaned_binary):
    th1, th2 = 30, 120
    edges = cv2.Canny(cleaned_binary, th1, th2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_contour = max(contours, key=cv2.contourArea)
    return cv2.boundingRect(max_contour)


def correct_perspective(cleaned_binary, x, y, w, h):
    warp_image = np.copy(cleaned_binary)
    rect = np.float32([[x, y], [x + w, y],
                       [x, y + h], [x + w, y + h]])
    output_size = 300
    dst_pts = np.float32([
        [0, 0], [output_size, 0],
        [0, output_size], [output_size, output_size]])
    perspective_matrix = cv2.getPerspectiveTransform(rect, dst_pts)
    corrected_perspective_image = cv2.warpPerspective(warp_image, perspective_matrix, (output_size, output_size))
    return  corrected_perspective_image

def detect_bullets(corrected_perspective_image):
    return

def assign_points(warped_image):
    return

if __name__ == '__main__':
    image = acquisition("target_images/clean_target.jpg")  #insert name_file to analyze

    preprocessed = preprocess_image(image)
    x, y, w, h = detect_target_contours(preprocessed)
    warped = correct_perspective(preprocessed, x, y, w, h)
    output_image = detect_bullets(warped)
    score = assign_points(output_image)

    #shows a unique window with input and analyzed image
    concat_images = np.concatenate((image, output_image), axis=1)
    cv2.imshow('Input vs Analyzed Target', concat_images)

    #press a random keyboard key to end the running main
    cv2.waitKey(0)
    cv2.destroyAllWindows()
