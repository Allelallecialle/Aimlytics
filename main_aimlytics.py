import numpy as np
import cv2

output_size = 300
target_center = (150, 150)
caliber_mm = 4.5
bullet_hole_area_mm = np.square((caliber_mm / 2)) * np.pi
pixels_per_mm = 300 / 45
expected_area_px = bullet_hole_area_mm * (pixels_per_mm ** 2)
max_score = 10
decimal_max_score = 10.9
target_radius_mm = 22.5
ring_width_mm = target_radius_mm / max_score
ring_radii_px = [(i * ring_width_mm) * pixels_per_mm for i in range(1, max_score + 1)]


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
    dst_pts = np.float32([
        [0, 0], [output_size, 0],
        [0, output_size], [output_size, output_size]])
    perspective_matrix = cv2.getPerspectiveTransform(rect, dst_pts)
    corrected_perspective_image = cv2.warpPerspective(warp_image, perspective_matrix, (output_size, output_size))
    return  corrected_perspective_image

def detect_bullets(corrected_perspective_image):
    blurred = cv2.GaussianBlur(corrected_perspective_image, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    best_contour = None
    min_diff = float('inf')

    for contour in contours:
        _ , radius = cv2.minEnclosingCircle(contour)
        area_px = np.pi * (radius ** 2)
        x, y, w, h = cv2.boundingRect(contour)
        if x < 5 or y < 5 or x + w > output_size - 5 or y + h > output_size - 5:
            continue

        if 0.2 * expected_area_px < area_px < expected_area_px * 1.3:
            diff = abs(area_px - expected_area_px)
            if diff < min_diff:
                min_diff = diff
                best_contour = contour

    output_image = cv2.cvtColor(corrected_perspective_image, cv2.COLOR_GRAY2BGR)
    if best_contour is None:
        hit_center = None
        return hit_center, output_image
    else:
        (x, y), radius = cv2.minEnclosingCircle(best_contour)
        hit_center = (int(x), int(y))
        cv2.circle(output_image, hit_center, int(radius), (0, 0, 255), 2)
        cv2.circle(output_image, hit_center, 2, (255, 255, 255), -1)
        return hit_center, output_image

def assign_points(hit_center):
    dx = hit_center[0] - target_center[0]
    dy = hit_center[1] - target_center[1]
    distance_px = np.sqrt(dx ** 2 + dy ** 2)

    score = 0
    for i, radius in enumerate(ring_radii_px):
        if distance_px <= radius:
            score = max_score - i
            break

    str_score = f"Score: {score} (distance from bullseye: {distance_px:.2f}px)"
    return str_score

def assign_points_decimal(hit_center):
    decimal_ring_width_mm = ring_width_mm / 10

    decimal_radii_mm = [i * decimal_ring_width_mm for i in range(110)]
    decimal_radii_px = [r * pixels_per_mm for r in decimal_radii_mm]

    dx = hit_center[0] - target_center[0]
    dy = hit_center[1] - target_center[1]
    distance_px = np.sqrt(dx ** 2 + dy ** 2)

    score = 0.0
    for i, radius in enumerate(decimal_radii_px):
        if distance_px <= radius:
            score = decimal_max_score - (i * 0.1)
            break

    str_score_decimal = f"Score: {score:.1f}"
    return str_score_decimal

if __name__ == '__main__':
    image = acquisition("target_images/photo_2025-03-20_09-50-51.jpg")  # Insert name_file to analyze

    preprocessed = preprocess_image(image)
    x, y, w, h = detect_target_contours(preprocessed)
    warped = correct_perspective(preprocessed, x, y, w, h)
    hit_cntr, output_image = detect_bullets(warped)
    if hit_cntr is not None:
        res_score = assign_points(hit_cntr)
        decimal_res_score = assign_points_decimal(hit_cntr)
        print(decimal_res_score)
        text = f"{decimal_res_score}"
        font = cv2.QT_FONT_NORMAL
        font_scale = 0.5
        color = (0, 0, 255)
        thickness = 1
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_offset_x = hit_cntr[0] + text_width // 3
        text_offset_y = hit_cntr[1] + text_height // 2
        cv2.putText(output_image, text, (text_offset_x, text_offset_y), font, font_scale, color, thickness)
    else:
        print("No hits detected")

    # Shows a unique window with input and analyzed image.
    # We need to reshape the output before the concat() because it's smaller due to the perspective transformation
    common_height = image.shape[0]
    scale_factor = common_height / output_image.shape[0]
    new_width = int(output_image.shape[1] * scale_factor)
    resized_output = cv2.resize(output_image, (new_width, common_height))

    concat_images = np.concatenate((image, resized_output), axis=1)

    # Create a resizable window
    cv2.namedWindow('Input vs Analyzed Target', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Input vs Analyzed Target', 1500, 900)
    cv2.imshow('Input vs Analyzed Target', concat_images)

    # Press a random keyboard key or the window's X to end the running main
    while True:
        key = cv2.waitKey(1)
        if cv2.getWindowProperty('Input vs Analyzed Target', cv2.WND_PROP_VISIBLE) < 1:
            break
        elif key != -1:
            break
    cv2.destroyAllWindows()
