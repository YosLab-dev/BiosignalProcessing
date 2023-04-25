from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import skimage.filters as skf
import cv2
import os
import skimage.measure
import pandas as pd
from skimage import io, color, measure, exposure
from scipy.stats import f_oneway
from scipy.stats import kruskal, mannwhitneyu
from sklearn.metrics import mean_absolute_error, mean_squared_error

input_images_path = "\Images"
imageNames = os.listdir(input_images_path)
vector = range(len(imageNames) - 1)
root = "Preliminary Results\\"
file = open("Results.txt", "w")

# In this global variable, the most relevant intensity values will be stored
intensityValues = np.zeros([3, 10])

def createPath():
    for i in vector:
        file = imageNames[i]

        # Path to create folder
        path = root + file
        print(path)
        directory = Path(path)
        directory.mkdir(parents=True)

def saveImage(image, path):
    result = cv2.imwrite(path, image)
    if result == False:
        print("Error in saving file! ):")
    cv2.destroyAllWindows()

def masked_image(image, mask):
    r = image[:,:,0] * mask
    g = image[:,:,1] * mask
    b = image[:,:,2] * mask
    return np.dstack([r,g,b])

def thresholdingMethods(imageGrayScale, imgOriginal, pathImage, nameImage):

    # Image filtering using threshold
    threshots = skf.threshold_otsu(imageGrayScale)
    threshYen = skf.threshold_yen(imageGrayScale)
    threshISO = skf.threshold_isodata(imageGrayScale)
    threshMin = skf.threshold_sauvola(imageGrayScale)

    # Applying thresholding
    otsu = imageGrayScale <= threshots
    yen = imageGrayScale <= threshYen
    iso = imageGrayScale <= threshISO
    sav = imageGrayScale <= threshMin

    # Applying the measurements to each labeled segment
    labeledOtsu = skimage.measure.label(otsu)
    labeledYen = skimage.measure.label(yen)
    labeledISO = skimage.measure.label(iso)
    labeledSav = skimage.measure.label(sav)

    # Converting the labels to a color image for each case
    labeledImgOt = color.label2rgb(labeledOtsu, image=imageGrayScale, bg_label=0)
    pathImOts = pathImage + "LabelsOtsu" + nameImage
    labeledImgY = color.label2rgb(labeledYen, image=imageGrayScale, bg_label=0)
    pathImYen = pathImage + "LabelsYen" + nameImage
    labeledImgISO = color.label2rgb(labeledISO, image=imageGrayScale, bg_label=0)
    pathImISO = pathImage + "LabelsISO" + nameImage
    labeledImgS = color.label2rgb(labeledSav, image=imageGrayScale, bg_label=0)
    pathImS = pathImage + "LabelsSav" + nameImage

    # Save labeled image for each case
    io.imsave(pathImOts, labeledImgOt)
    io.imsave(pathImYen, labeledImgY)
    io.imsave(pathImISO, labeledImgISO)
    io.imsave(pathImS, labeledImgS)

    # Calculating the properties of regions
    propsOt = skimage.measure.regionprops_table(labeledOtsu, imageGrayScale, properties = ['label', 'area', 'perimeter', 'eccentricity', 'solidity'])
    propsYen = skimage.measure.regionprops_table(labeledYen, imageGrayScale, properties=['label', 'area', 'perimeter', 'eccentricity', 'solidity'])
    propsISO = skimage.measure.regionprops_table(labeledISO, imageGrayScale, properties=['label', 'area', 'perimeter', 'eccentricity', 'solidity'])
    #propsSav = skimage.measure.regionprops_table(labeledSav, imageGrayScale, properties=['label', 'area', 'perimeter', 'eccentricity', 'solidity'])

    # Creating all the dataframes associated to props calculated before
    dfOtsu = pd.DataFrame(propsOt)
    dfYen = pd.DataFrame(propsYen)
    dfISO = pd.DataFrame(propsISO)

    # Saving all the results in csv file
    df1 = pd.concat([dfOtsu, dfYen])
    dataframe = pd.concat([df1, dfISO])
    path = pathImage + "PropsThresholding.csv"
    dataframe.to_csv(path)

    # Applying masking between thresholding and the original image
    filtered = masked_image(imgOriginal, otsu)

    # Saving the original segmented image
    pathImOts = pathImage + "OtsuTh" + nameImage
    saveImage(filtered, pathImOts)

    # Applying the Watershed Algorithm under the Otsu threshold
    WatershedAlg(imageGrayScale, imgOriginal, threshots, pathImage, nameImage, "Otsu")
    pathH = pathImage + "HistogramOtsuTh" + nameImage + ".png"
    imageThr = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)

    # The intensity values will be stored in an array (associated with the image)
    createHistogram(imageThr, pathImage, 0, pathH)

    # ------ In this part, the aforementioned methodology is repeated using the different thresholds -----

    filtered = masked_image(imgOriginal, yen)
    pathImYen = pathImage + "YenTh" + nameImage
    saveImage(filtered, pathImYen)
    WatershedAlg(imageGrayScale, imgOriginal, threshYen, pathImage, nameImage, "Yen")
    pathH = pathImage + "HistogramYenTh" + nameImage + ".png"
    imageThr = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    createHistogram(imageThr, pathImage, 1, pathH)

    filtered = masked_image(imgOriginal, iso)
    pathImISO = pathImage + "ISOTh" + nameImage
    saveImage(filtered, pathImISO)
    WatershedAlg(imageGrayScale, imgOriginal, threshISO, pathImage, nameImage, "ISO")
    pathH = pathImage + "HistogramISOTh" + nameImage + ".png"
    imageThr = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    createHistogram(imageThr, pathImage, 2, pathH)

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

def createHistogram(image, pathCSV, index, pathHistogram):

    # Applying the mask predefined
    mskdIm = cv2.bitwise_and(image, image, mask=mask)

    # Calculating the normalized histogram
    hist, bins = exposure.histogram(mskdIm, nbins=256, normalize=True)
    plt.plot(bins[1:], hist[1:])

    # Saving the histogram results
    plt.savefig(pathHistogram, dpi=300)
    plt.close()

    # Reorder the histogram in descending order
    sortedHistogram = np.sort(hist, axis=None)[::-1]

    # Creating the vector that will store the five most relevant intensities
    for i in range (1, 11, 1):
        intensityValues[index, i-1] = sortedHistogram[i]

    columnsLabels = ['Intensity(Max)', 'Intensity(Max-1)', 'Intensity(Max-2)', 'Intensity(Max-3)', 'Intensity(Max-4)', 'Intensity(Max-5)', 'Intensity(Max-6)', 'Intensity(Max-7)', 'Intensity(Max-8)', 'Intensity(Max-9)']
    df = pd.DataFrame(intensityValues, columns=columnsLabels)
    pathCSV = pathCSV + "\\" + "IntensityValues.csv"
    df.to_csv(pathCSV)


def analysisANOVA(pathCSV):

    df = pd.read_csv(pathCSV)

    listV = df.values.tolist()
    array = np.array(listV)
    arrSplit = np.array_split(array, 3)

    # Looping through the list and removing the first column of each array
    for i in range(len(arrSplit)):
        arrSplit[i] = np.delete(arrSplit[i], 0, axis=1)

    yen = np.zeros(5)
    isodata = np.zeros(5)
    otsu = np.zeros(5)

    # Matrix of segmented image intensities for each method
    otsu = arrSplit[0]
    yen = arrSplit[1]
    isodata = arrSplit[2]

    #
    file.write("*************************************************\n")
    file.write(pathCSV+'\n')
    file.write("*************************************************\n")
    u_stat, p_value = mannwhitneyu(otsu, yen)
    file.write("------------ Otsu - Yen ------------\n")
    file.write(np.array2string(u_stat))
    file.write("\n")
    file.write(np.array2string(p_value))
    file.write("\n")
    file.write("------------------------------------\n")

    u_stat, p_value = mannwhitneyu(yen, isodata)
    file.write("------------ Yen - Isodata ------------\n")
    file.write(np.array2string(u_stat))
    file.write("\n")
    file.write(np.array2string(p_value))
    file.write("\n")
    file.write("------------------------------------\n")

    u_stat, p_value = mannwhitneyu(otsu, isodata)
    file.write("------------ Otsu - Isodata ------------\n")
    file.write(np.array2string(u_stat))
    file.write("\n")
    file.write(np.array2string(p_value))
    file.write("\n")
    file.write("------------------------------------\n")

    # Calculate the MAE between the vectors "yen" and "otsu"
    mae_yen_otsu = mean_absolute_error(yen, otsu)
    file.write("MAE entre yen y otsu: ")
    file.write(np.array2string(mae_yen_otsu))
    file.write("\n")

    # Calculate the MSE between the vectors "yen" and "otsu"
    mse_yen_otsu = mean_squared_error(yen, otsu)
    file.write("MSE entre yen y otsu: ")
    file.write(np.array2string(mse_yen_otsu))
    file.write("\n")

    # Calculate the MAE between the vectors "yen" and "isodata"
    mae_yen_isodata = mean_absolute_error(yen, isodata)
    file.write("MAE entre yen y isodata: ")
    file.write(np.array2string(mae_yen_isodata))
    file.write("\n")

    # Calculate the MSE between the vectors "yen" and "isodata"
    mse_yen_isodata = mean_squared_error(yen, isodata)
    file.write("MSE entre yen y isodata: ")
    file.write(np.array2string(mse_yen_isodata))
    file.write("\n")

    # Calculate the MAE between the vectors "otsu" and "isodata"
    mae_otsu_isodata = mean_absolute_error(otsu, isodata)
    file.write("MAE entre otsu e isodata: ")
    file.write(np.array2string(mae_otsu_isodata))
    file.write("\n")

    # Calculate the MSE between the vectors "otsu" and "isodata"
    mse_otsu_isodata = mean_squared_error(otsu, isodata)
    file.write("MSE entre otsu e isodata: ")
    file.write(np.array2string(mse_otsu_isodata))
    file.write("\n")


# ----------- M A I N P R O G R A M -----------

#createPath()

for i in vector:
    pathImage = "\Images\\" + imageNames[i]
    pathMeth = root + "\\" + imageNames[i] + "\\"
    pathOriginal = root + "\\" + imageNames[i] + "\\" + "Original" + imageNames[i]
    pathFilter = root + "\\" + imageNames[i] + "\\" + "BilateralF" + imageNames[i]
    pathEq = root + "\\" + imageNames[i] + "\\" + "EquIm" + imageNames[i]
    pathMask = root + "\\" + "Mask.png"
    pathMasked = root + "\\" + imageNames[i] + "\\" + "Masked" + imageNames[i]
    imgOriginal = cv2.imread(pathImage)

    # Applying bilateral filter
    imgFiltered = cv2.bilateralFilter(imgOriginal, 2, 15, 15)

    # Creating the mask to remove the black regiones in the image
    mask = cv2.imread(pathMask, 0)
    imgFiltMsk = cv2.bitwise_and(imgFiltered, imgFiltered, mask=mask)

    # Saving the results
    saveImage(imgOriginal,pathOriginal)
    saveImage(imgFiltMsk,pathFilter)

    # Reading the filtered and masked image into gray scale
    segmentGray = cv2.cvtColor(imgFiltMsk, cv2.COLOR_BGR2GRAY)
    thresholdingMethods(segmentGray, imgFiltMsk, pathMeth, imageNames[i])

    # Reading
    pathCSV = pathMeth + "\\" + "IntensityValues.csv"
    analysisANOVA(pathCSV)
file.close() 
