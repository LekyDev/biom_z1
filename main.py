import cv2
import numpy as np
import helpMethods as myFunctions

cv2.namedWindow(myFunctions.title_window, cv2.WINDOW_NORMAL)
key = cv2.waitKey(0)

myFunctions.createGUI(0, 23, 6, 27, 28, 11, 15, 23, 21, 32, 152)
groundTruthZrenicka = [156, 118, 56]
groundTruthDuhovka = [156, 110, 111]
img = cv2.imread('duhovky/001/L/S1001L01.jpg', 0)

#myFunctions.createGUI(0, 23, 6, 27, 28, 17, 15, 38, 21, 32, 152)
#groundTruthZrenicka = [158, 176, 38]
#groundTruthDuhovka = [166, 180, 108]
#img = cv2.imread('duhovky/020/L/S1020L01.jpg', 0)

#myFunctions.createGUI(1, 5, 19, 24, 28, 25, 33, 121, 66, 32, 79)
#groundTruthZrenicka = [168, 162, 32]
#groundTruthDuhovka = [166, 156, 105]
#img = cv2.imread('duhovky/047/R/S1047R01.jpg', 0)

while True:
    equalize = cv2.getTrackbarPos('Hist ekv', myFunctions.title_window)
    gcSize = cv2.getTrackbarPos("Gauss Blur - Size", myFunctions.title_window)
    gcSigma = cv2.getTrackbarPos("Gauss Blur - Sigma", myFunctions.title_window)
    threshold1 = cv2.getTrackbarPos("threshold1", myFunctions.title_window)
    threshold2 = cv2.getTrackbarPos("threshold2", myFunctions.title_window)
    accumulator = cv2.getTrackbarPos("Accumulator", myFunctions.title_window)
    minDist = cv2.getTrackbarPos("minDist", myFunctions.title_window)
    param1 = cv2.getTrackbarPos("param1", myFunctions.title_window)
    param2 = cv2.getTrackbarPos("param2", myFunctions.title_window)
    minRadius = cv2.getTrackbarPos("minRadius", myFunctions.title_window)
    maxRadius = cv2.getTrackbarPos("maxRadius", myFunctions.title_window)

    # zmensime akumulator
    accumulator = accumulator / 10

    # ak by bola 0, tak padne program
    if param2 == 0:
        param2 = 1

    if param1 == 0:
        param1 = 1

    if minDist == 0:
        minDist = 1

    # size a sigma nesmie byt parna
    if (gcSize % 2 == 0):
        gcSize = gcSize + 1


    # filtre
    if equalize:
        img = cv2.equalizeHist(img)

    imgAfterGaussianBlur = cv2.GaussianBlur(img, (gcSize, gcSize), gcSigma, borderType=cv2.BORDER_DEFAULT)

    imgAfterCanny = cv2.Canny(imgAfterGaussianBlur, threshold1, threshold2)

    imgAfterCvtColor = cv2.cvtColor(imgAfterGaussianBlur, cv2.COLOR_GRAY2BGR)

    # funkcia Hough najde kruhy
    circles = cv2.HoughCircles(imgAfterGaussianBlur, cv2.HOUGH_GRADIENT, dp=accumulator, minDist=minDist,
                               param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

    numberOfDuhovkas = 0
    numberOfZrenickas = 0
    arrayOfDuhovkas = []
    arrayOfZrenickas = []

    # ak je nejaky kruh
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # pozrieme či kruh je duhovka alebo zrenicka podla IoU a groundTruth
            if (myFunctions.DuhovkaOrZrenicka(i, groundTruthDuhovka, groundTruthZrenicka)) == 0:  # Is Duhovka
                # mame threshold 0.75, dal by sa nastavit acceptance
                if myFunctions.getIoU(i, groundTruthDuhovka) > 0.75:
                    numberOfDuhovkas += 1
                    arrayOfDuhovkas.append(i)
                    iouDuhovka = myFunctions.getIoU(i, groundTruthDuhovka)
                    # nakresli kruh duhovky
                    cv2.circle(imgAfterCvtColor, (i[0], i[1]), i[2], (255, 128, 0), 2)

            elif (myFunctions.DuhovkaOrZrenicka(i, groundTruthDuhovka, groundTruthZrenicka)) == 1:  # Is Zrenicka
                if myFunctions.getIoU(i, groundTruthZrenicka) > 0.75:
                    numberOfZrenickas += 1
                    arrayOfZrenickas.append(i)
                    iouZrenicka = myFunctions.getIoU(i, groundTruthZrenicka)
                    # nakresli kruh zrenicky
                    cv2.circle(imgAfterCvtColor, (i[0], i[1]), i[2], (255, 153, 255), 2)

    duhovkaFN = 0
    zrenickaFN = 0
    # Checking TP and FP values
    if numberOfDuhovkas > 0:
        numberTPDuhovka = 1
        tpDuhovka = "TP-DUHOVKA: 1"
        fpDuhovka = 'FP-DUHOVKA: '+ str(numberOfDuhovkas - 1)
    elif numberOfDuhovkas == 0:
        tpDuhovka = "TP-DUHOVKA: 0"
        fpDuhovka = "FP-DUHOVKA: 0"
        numberTPDuhovka = 0

    if numberOfZrenickas > 0:
        numberTPZrenicka = 1
        tpZrenicka = 'TP-ZRENICKA: 1'
        fpZrenicka = 'FP-ZRENICKA: ' + str(numberOfZrenickas - 1)
    elif numberOfZrenickas == 0:
        tpZrenicka = 'TP-ZRENICKA: 0'
        fpZrenicka = 'FP-ZRENICKA: 0'
        numberTPZrenicka = 0

    if len(arrayOfDuhovkas) > 0 :
        iouDuhovka = "iouDuhovka - " + str(round(iouDuhovka, 2))

    if len(arrayOfZrenickas) > 0:
        iouZrenicka = "iouZrenicka - " + str(round(iouZrenicka, 2))


    # ak je TP 1, tak sme nasli duhovku/zrenicku a prislusne FN musi byt 0 (nie je ziadny nedetekovany objekt, ktory tam je)
    if numberTPDuhovka == 0:
        duhovkaFN=1
    else:
        duhovkaFN=0
    if numberTPZrenicka == 0:
        zrenickaFN = 1
    else:
        zrenickaFN = 0

    recall = (numberTPDuhovka + numberTPZrenicka) / ((numberTPDuhovka + numberTPZrenicka) + (duhovkaFN + zrenickaFN))

    if numberOfDuhovkas > 0 and numberOfZrenickas > 0:
        precision = (numberTPDuhovka + numberTPZrenicka) / ((numberTPDuhovka + numberTPZrenicka) + ((numberOfDuhovkas - 1)+(numberOfZrenickas - 1)))
        print(str(precision) + " precision")
        print(str(recall) + " recall")
    elif(numberOfDuhovkas==0 and numberOfZrenickas > 0):
        precision = (0 + numberTPZrenicka) / ((0 + numberTPZrenicka) + (0 + (numberOfZrenickas - 1)))
        print(str(precision) + " precision")
        print(str(recall) + " recall")
    elif numberOfDuhovkas > 0 and numberOfZrenickas == 0:
        precision = (numberTPDuhovka + 0) / ((numberTPDuhovka + 0) + ((numberOfDuhovkas - 1) + 0))
        print(str(precision) + " precision")
        print(str(recall) + " recall")
    else:
        print(str(0.0) + " precision")
        print(str(recall) + " recall")

    # vypis na obrazovku
    imgAfterCvtColor = cv2.putText(imgAfterCvtColor, tpZrenicka, (00, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1, cv2.LINE_AA, False)
    imgAfterCvtColor = cv2.putText(imgAfterCvtColor, fpZrenicka, (00, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1, cv2.LINE_AA, False)
    imgAfterCvtColor = cv2.putText(imgAfterCvtColor, tpDuhovka, (00, 10), cv2.FONT_HERSHEY_SIMPLEX , 0.3, (0,0,255), 1, cv2.LINE_AA, False)
    imgAfterCvtColor = cv2.putText(imgAfterCvtColor, fpDuhovka, (00, 20), cv2.FONT_HERSHEY_SIMPLEX , 0.3, (0,0,255), 1, cv2.LINE_AA, False)

    if len(arrayOfDuhovkas) > 0:
        imgAfterCvtColor = cv2.putText(imgAfterCvtColor, iouDuhovka, (00, 60), cv2.FONT_HERSHEY_SIMPLEX , 0.3, (0,0,255), 1, cv2.LINE_AA, False)
    if len(arrayOfZrenickas) > 0:
        imgAfterCvtColor = cv2.putText(imgAfterCvtColor, iouZrenicka, (00, 70), cv2.FONT_HERSHEY_SIMPLEX , 0.3, (0,0,255), 1, cv2.LINE_AA, False)

    # ukoncenie programu
    k = cv2.waitKey(1)
    if k == 27:
        break

    cv2.imshow("Oko", imgAfterCvtColor)

    # BONUS
    if numberOfDuhovkas == 1 and numberOfZrenickas == 1:
         cv2.imshow("Segmentovany obrazok", myFunctions.segmentation(imgAfterCvtColor, arrayOfDuhovkas[0], arrayOfZrenickas[0]))