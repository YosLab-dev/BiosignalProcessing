from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import skimage.filters as skf
import cv2
import mplhep as hep
import os

input_images_path = "\Images"
imageNames = os.listdir(input_images_path)
vector = range(len(imageNames) - 1)
root = "Preliminary Results\\"

def createPath():
    for i in vector:
        file = imageNames[i]
        path = root + file #Path to create folder
        print(path)
        directory = Path(path)
        directory.mkdir(parents=True)

def saveImage(image, path):
    result = cv2.imwrite(path, image)
    if result == True:
        print("File saved successfully!")
    else:
        print("Error in saving file")

def masked_image(image, mask):
    r = image[:,:,0] * mask
    g = image[:,:,1] * mask
    b = image[:,:,2] * mask
    return np.dstack([r,g,b])

def thresholdingMethods(imageGrayScale, imageOriginal, pathImage, nameImage):
    threshots = skf.threshold_otsu(imageGrayScale)
    threshYen = skf.threshold_yen(imageGrayScale)
    threshISO = skf.threshold_isodata(imageGrayScale)
    threshMin = skf.threshold_sauvola(imageGrayScale)

    otsu = imageGrayScale <= threshots
    yen = imageGrayScale <= threshYen
    iso = imageGrayScale <= threshISO
    sav = imageGrayScale <= threshMin

    filtered = masked_image(imageOriginal, otsu)
    pathIm = pathImage + "OtsuTh" + nameImage
    #WatershedAlg(imageGrayScale, imageOriginal, thresholdValue, path, nameImage, type)
    WatershedAlg(imageGrayScale, imageOriginal, threshots, pathImage, nameImage, "Otsu")
    saveImage(filtered, pathIm)
    filtered = masked_image(imageOriginal, yen)
    pathIm = pathImage + "YenTh" + nameImage
    WatershedAlg(imageGrayScale, imageOriginal, threshYen, pathImage, nameImage, "Yen")
    saveImage(filtered, pathIm)
    filtered = masked_image(imageOriginal, iso)
    pathIm = pathImage + "ISOTh" + nameImage
    WatershedAlg(imageGrayScale, imageOriginal, threshISO, pathImage, nameImage, "ISO")
    saveImage(filtered, pathIm)
    filtered = masked_image(imageOriginal, sav)
    pathIm = pathImage + "SavTh" + nameImage
    saveImage(filtered, pathIm)

def WatershedAlg(imageGrayScale, imageOriginal, thresholdValue, path, nameImage, type):

    pathIm = path + "BordersWs" + type + nameImage
    gray_img = imageGrayScale

    # Using 5*5 kernel
    median_filtered = cv2.medianBlur(gray_img, 5)

    # 3*3 Sobel Filters
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img_sobelx = cv2.Sobel(median_filtered, cv2.CV_8U, 1, 0, ksize=3)
    img_sobely = cv2.Sobel(median_filtered, cv2.CV_8U, 0, 1, ksize=3)

    # Adding mask to the image
    img_sobel = img_sobelx + img_sobely + gray_img

    # Set threshold and maxValue
    threshold = thresholdValue
    maxValue = 255

    # Threshold the pixel values
    th, thresh = cv2.threshold(img_sobel, threshold, maxValue, cv2.THRESH_BINARY)

    # To remove any small white noises in the image using morphological opening.
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Dilation increases object boundary to background.
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    #  White region shows sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Watershed algorithm
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    contours, hierarchy = cv2.findContours(sure_fg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Creating a numpy array for markers and converting the image to 32 bit using dtype paramter
    marker = np.zeros((gray_img.shape[0], gray_img.shape[1]), dtype=np.int32)

    marker = np.int32(sure_fg) + np.int32(sure_bg)

    # Marker Labelling
    for id in range(len(contours)):
        cv2.drawContours(marker, contours, id, id + 2, -1)

    marker = marker + 1
    marker[unknown == 255] = 0
    copy_img = imageOriginal.copy()
    cv2.watershed(copy_img, marker)
    copy_img[marker == -1] = (0, 0, 255)
    saveImage(copy_img, pathIm)




createPath()

for i in vector:
    pathImage = "\Images\\" + imageNames[i]
    segmentOriginal = imread(pathImage)
    segmentGray = cv2.cvtColor(segmentOriginal, cv2.COLOR_BGR2GRAY)
    path = root + "\\" + imageNames[i] + "\\" + imageNames[i]
    pathG = root + "\\" + imageNames[i] + "\\" + "Gray" + imageNames[i]
    pathMeth = root + "\\" + imageNames[i] + "\\"
    saveImage(segmentOriginal, path)
    saveImage(segmentGray, pathG)
    thresholdingMethods(segmentGray, segmentOriginal, pathMeth, imageNames[i])





