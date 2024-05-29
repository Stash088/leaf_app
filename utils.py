import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline
from skimage import measure, color
from skimage.feature import hessian_matrix , hessian_matrix_eigvals
from skimage.color import rgb2gray
from skimage.segmentation import clear_border
from skimage.filters import threshold_otsu
from scipy import ndimage as ndi
from sklearn.cluster import k_means
from scipy import optimize
def mask_leaf(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([15, 50, 50])
    upper_green = np.array([85, 255, 255])
    lower_brown = np.array([5, 50, 50])
    upper_brown = np.array([25, 255, 255])
    lower_yellow = np.array([20, 50, 50])
    upper_yellow = np.array([40, 255, 255])
    kernel = np.ones((7, 7), np.uint8)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask = cv2.bitwise_or(mask_green, mask_brown)
    mask = cv2.bitwise_or(mask, mask_yellow)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(mask)
    cv2.drawContours(mask, contours, -1, (255), 2)
    return mask


def venation_leaf(image):
    mask = mask_leaf(image)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ellipses = []
    for contour in contours:
        if len(contour) > 5:
            ellipse = cv2.fitEllipse(contour)
            ellipses.append(ellipse)
    venation = ""
    for ellipse in ellipses:
        angle = ellipse[2]
        if 45 <= angle <= 135:
            venation = "Параллельная"
        elif -45 >= angle >= -135:
            venation = "Перпендикулярная"
        else:
            venation = "Другая венация"
    return venation


def classfier_leaf(image):
    mask = mask_leaf(image)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    approx_contours = []
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        approx_contours.append(approx)
    leaf_shape = ""
    for contour in approx_contours:
        vertices = len(contour)
    if vertices == 3:
        leaf_shape = "Треугольная"
    elif vertices == 4:
        leaf_shape = "Прямоугольная или квадратная"
    elif vertices == 5:
        leaf_shape = "Пятиугольная"
    else:
        leaf_shape = "Другая форма"
    return leaf_shape


def measure_extract(image):
    if isinstance(image, str):
        try:
            image = cv2.imread(image)
        except Exception as e:
            print("Error loading image:", str(e))
            return None, None, None, None

    if image is None:
        print("Invalid image")
        return None, None, None, None

    mask = mask_leaf(image)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        print("No leaf contours found")
        return None, None, None, None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largest_contour = contours[0]

    x, y, w, h = cv2.boundingRect(largest_contour)
    top = (x + w // 2, y)
    bottom = (x + w // 2, y + h - 1)

    leftmost = tuple(largest_contour[largest_contour[:, :, 0].argmin()][0])
    rightmost = tuple(largest_contour[largest_contour[:, :, 0].argmax()][0])

    length = abs(bottom[1] - top[1])
    width = abs(rightmost[0] - leftmost[0])
    M = cv2.moments(largest_contour)
    center_x = int(M['m10'] / M['m00'])
    center_y = int(M['m01'] / M['m00'])

    distances_x = []
    distances_y = []
    for point in largest_contour:
        distance_x = point[0][0] - center_x
        distance_y = point[0][1] - center_y
        distances_x.append(distance_x)
        distances_y.append(distance_y)

    abs_diff_x = np.abs(distances_x)
    abs_diff_y = np.abs(distances_y)

    mean_abs_diff_x = np.mean(abs_diff_x)
    mean_abs_diff_y = np.mean(abs_diff_y)

    fluctuating_asymmetry = (mean_abs_diff_x / mean_abs_diff_y) if mean_abs_diff_y != 0 else 1
    return length, width, fluctuating_asymmetry


def fractal_dimension(image):
    mask = mask_leaf(image)

    # Нахождение контуров объектов на изображении
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Вычисляем фрактальную размерность с помощью бокс-счета
    box_count = 0
    box_size = 2
    while box_size < image.shape[0]:
        for contour in contours:
            for point in contour:
                if point[0][0] % box_size == 0 and point[0][1] % box_size == 0:
                    box_count += 1
                    break
        box_size *= 2

    # Вычисляем фрактальную размерность
    fractal_dimension = np.log(box_count) / np.log(box_size)

    return fractal_dimension


def fractal_spectrum(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Преобразование Фурье изображения
    f = np.fft.fftshift(np.fft.fft2(image))

    # Расчет модуля спектра
    spectrum = np.abs(f)

    # Логарифмическое преобразование
    spectrum = np.log(1 + spectrum)
    spectrum = (spectrum - np.min(spectrum)) / (np.max(spectrum) - np.min(spectrum))

    return spectrum


def fractal_texture_index(image):
    mask = mask_leaf(image)

    f = np.fft.fftshift(np.fft.fft2(image))

    # Расчет модуля спектра
    spectrum = np.abs(f)

    # Логарифмическое преобразование
    spectrum = np.log(1 + spectrum)

    # Вычисление суммы интенсивностей пикселей в спектре
    total_intensity = np.sum(spectrum)

    # Вычисление суммы квадратов интенсивностей пикселей в спектре
    squared_intensity = np.sum(spectrum ** 2)

    # Вычисление фрактального индекса текстуры
    texture_index = squared_intensity / total_intensity

    return texture_index

def fractal_sphericity(image):
    mask = mask_leaf(image)
    # Вычисление периметра
    perimeter = np.sum(mask[1:] != mask[:-1]) + np.sum(mask[:, 1:] != mask[:, :-1]) + mask[0, 0] + mask[-1, -1]

    # Вычисление площади
    area = np.sum(mask)

    # Вычисление фрактальной размерности
    fractal_dimension = np.log(perimeter) / np.log(area)

    # Вычисление фрактальной сферичности
    fractal_sphericity = area / (perimeter ** 2)

    return fractal_sphericity

def calculate_leaf_dimensions(img):
    def calc_px_per_cm2(resolution):
        return int((resolution ** 2) / (2.54 ** 2))

    def find_leaves(binary_img, min_size):
        labels, _ = ndi.label(binary_img)
        items, area = np.unique(labels, return_counts=True)
        big_items = items[area > min_size][1:]  # subtract background
        leaf = np.isin(labels, big_items)  # keep items that are leaf
        item_areas = area[np.isin(items, big_items)]
        item_bboxes = ndi.find_objects(labels == big_items[:, None])[1:]  # subtract background
        return leaf, item_areas, item_bboxes

    if img.shape[2] == 4:
        # Convert RGBA image to RGB
        img = img[:, :, :3]

    # Convert image to grayscale
    img_gray = rgb2gray(img)
    resolution = 300  # Resolution of the image in dots per inch
    min_size = 1000  # Minimum size of a leaf in pixels

    # Convert image to binary
    thresh = threshold_otsu(img_gray)
    binary = img_gray < thresh
    binary = clear_border(binary)

    # Find leaves and calculate statistics
    leaf, item_areas, item_bboxes = find_leaves(binary, min_size)
    px_per_cm2 = calc_px_per_cm2(resolution)
    num_leaves = len(item_areas)

    # Calculate length and width of each leaf
    leaf_lengths = []
    leaf_widths = []
    for bbox in item_bboxes:
        min_row, max_row, min_col, max_col = bbox
        leaf_lengths.append((max_row - min_row) / px_per_cm2)
        leaf_widths.append((max_col - min_col) / px_per_cm2)

    return leaf_lengths, leaf_widths

def calculate_area_leafs(img):
    def calc_px_per_cm2(resolution):
        return int((resolution ** 2) / (2.54 ** 2))

    def find_leaves(binary_img, min_size):
        labels, _ = ndi.label(binary_img)
        items, area = np.unique(labels, return_counts=True)
        big_items = items[area > min_size][1:]  # subtract background
        leaf = np.isin(labels, big_items)  # keep items that are leaf
        item_areas = area[np.isin(items, big_items)]
        return leaf, item_areas

    if img.shape[2] == 4:
        # Convert RGBA image to RGB
        img = img[:, :, :3]

    # Convert image to grayscale
    img_gray = rgb2gray(img)
    resolution = 300  # Resolution of the image in dots per inch
    min_size = 1000  # Minimum size of a leaf in pixels

    # Convert image to binary
    thresh = threshold_otsu(img_gray)
    binary = img_gray < thresh
    binary = clear_border(binary)

    # Find leaves and calculate statistics
    leaf, item_areas = find_leaves(binary, min_size)
    px_per_cm2 = calc_px_per_cm2(resolution)
    num_leaves = len(item_areas)
    avg_area = np.mean(item_areas) / px_per_cm2
    total_area = np.sum(item_areas) / px_per_cm2
    return total_area

def calculate_chlorophyll_index(img):
    # Преобразование изображения в пространство цветности RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Разделение каналов изображения
    red_channel = img_rgb[:, :, 0]
    green_channel = img_rgb[:, :, 1]
    blue_channel = img_rgb[:, :, 2]

    # Расчет спектральных индексов хлорофилла
    chla = (green_channel - red_channel) / (green_channel + red_channel)
    chlb = (blue_channel - red_channel) / (blue_channel + red_channel)
    chla_b = chla / chlb
    return chla, chlb, chla_b


def canny_edges(image):
    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(10, 10))
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    mask = mask_leaf(image)
    img_equalized = clahe.apply(blurred)

    # Применить адаптивное пороговое преобразование
    edge_equalized = cv2.adaptiveThreshold(img_equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    edge_equalized = cv2.dilate(edge_equalized, kernel)
    edge_equalized = cv2.erode(edge_equalized, kernel)
    edge_equalized = cv2.morphologyEx(edge_equalized, cv2.MORPH_CLOSE, kernel)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edge_equalized = cv2.morphologyEx(edge_equalized, cv2.MORPH_CLOSE, kernel2)

    _, threshold = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY_INV)
    result = cv2.bitwise_and(edge_equalized, threshold)
    result = cv2.medianBlur(result, 3)

    return result

def ndvi_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    red = image[:,:,0].astype(np.float32)
    nir = image[:,:,2].astype(np.float32)
    ndvi = (nir - red) / (nir + red)

    plt.imshow(ndvi, cmap='RdYlGn')
    plt.colorbar()
    plt.show()
    return ndvi

def calculate_shapes(img):
    def calc_px_per_cm2(resolution):
        return int((resolution ** 2) / (2.54 ** 2))
    def find_leaves(binary_img, min_size):
        labels, _ = ndi.label(binary_img)
        items, area = np.unique(labels, return_counts=True)
        big_items = items[area > min_size][1:]  # subtract background
        leaf = np.isin(labels, big_items)  # keep items that are leaf
        item_areas = area[np.isin(items, big_items)]
        return leaf, item_areas
    if img.shape[2] == 4:
        # Convert RGBA image to RGB
        img = img[:, :, :3]
    # Convert image to grayscale
    img_gray = rgb2gray(img)
    resolution = 300  # Resolution of the image in dots per inch
    min_size = 1000  # Minimum size of a leaf in pixels
    # Convert image to binary
    thresh = threshold_otsu(img_gray)
    binary = img_gray < thresh
    binary = clear_border(binary)
    # Find leaves and calculate statistics
    leaf, item_areas = find_leaves(binary, min_size)
    px_per_cm2 = calc_px_per_cm2(resolution)
    # Calculate width and height of the image
    width = img.shape[1] / px_per_cm2
    height = img.shape[0] / px_per_cm2

    return width, height



# image = cv2.imread('data/leaf.png')

# skel = canny_edges(image)
# prun = prune(skel,min_length=1000)
# plt.imshow(prun) 
  
# # display that image 
# plt.show() 