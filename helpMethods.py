import cv2, math

title_window = 'Nastavenie parametrov'

# callback
def nothing(x):
    pass

def createGUI(histEkv, gaussianSize, gaussianSigma, canny1, canny2, accumulator, minDist, param1, param2, minRadius, maxRadius):
    cv2.createTrackbar('Hist ekv', title_window, histEkv, 1, nothing)
    cv2.createTrackbar("Gauss Blur - Size", title_window, gaussianSize, 100, nothing)
    cv2.createTrackbar("Gauss Blur - Sigma", title_window, gaussianSigma, 100, nothing)
    cv2.createTrackbar("threshold1", title_window, canny1, 100, nothing)
    cv2.createTrackbar("threshold2", title_window, canny2, 300, nothing)
    cv2.createTrackbar("Accumulator", title_window, accumulator, 60, nothing)
    cv2.createTrackbar("minDist", title_window, minDist, 60, nothing)
    cv2.createTrackbar("param1", title_window, param1, 170, nothing)
    cv2.createTrackbar("param2", title_window, param2, 110, nothing)
    cv2.createTrackbar("minRadius", title_window, minRadius, 90, nothing)
    cv2.createTrackbar("maxRadius", title_window, maxRadius, 300, nothing)

# vrati plochu zjednotenia 2 kruhov
def getUnion(circle1, circle2):
    intersection = getOverlap(circle1, circle2)
    areaCircle1 = math.pi * math.pow(circle1[2], 2)
    areaCircle2 = math.pi * math.pow(circle2[2], 2)
    return areaCircle1 + areaCircle2 - intersection


# vrati prekryv 2 kruhov
def getOverlap(circle1, circle2):
    d = math.sqrt(math.pow(circle1[0] - circle2[0], 2) + math.pow(circle1[1] - circle2[1], 2)) # vzdialenost dvoch kruznic

    if d < (circle1[2] - circle2[2]): # kruh 2 je v kruhu 1
        return math.pi * math.pow(circle2[2], 2)

    if d < (circle2[2] - circle1[2]): # kruh 1 je v kruhu 2
        return math.pi * math.pow(circle1[2], 2)

    if d > (circle1[2] + circle2[2]): # su rovnake
        return math.pi * math.pow(circle1[2], 2)

    if d < (circle1[2] + circle2[2]): # krizuju sa v 2 bodoch
        d1 = (math.pow(circle1[2], 2) - math.pow(circle2[2], 2) + math.pow(d, 2)) / (2 * d)
        d2 = d - d1
        intersection = (math.pow(circle1[2], 2)) * math.acos(d1 / circle1[2]) - d1 * math.sqrt(math.pow(circle1[2], 2) - math.pow(d1, 2)) + (math.pow(circle2[2], 2)) * math.acos(d2 / circle2[2]) - d2 * math.sqrt(math.pow(circle2[2], 2) - math.pow(d2, 2))
        return intersection

    return 0 # ziadny kontakt

# zisti či je kruh zrenicka alebo duhovka
def DuhovkaOrZrenicka(circle, groundtruthduhovka, groundtruthzrenicka):
    if getIoU(circle, groundtruthduhovka) > getIoU(circle, groundtruthzrenicka):
        return 0  # kruh je duhovka
    else:
        return 1  # kruh je zrenicka

# vrati intersection over union
def getIoU(circle1, circle2):
    return getOverlap(circle1, circle2) / getUnion(circle1, circle2)

# bonusova uloha
def segmentation(image, duhovka, zrenicka):
    h, w, ch = image.shape
    for y in range(0, h):
        for x in range(0, w):
            d = math.sqrt(math.pow(x - duhovka[0], 2) + math.pow(y - duhovka[1], 2))
            if d > duhovka[2]:
                image[y, x] = (0, 0, 0)
            else:
                image[y, x] = (255, 255, 255)

    for u in range(0, h):
        for z in range(0, w):
            d = math.sqrt(math.pow(z - zrenicka[0], 2) + math.pow(u - zrenicka[1], 2))
            if d < zrenicka[2]:
                image[u, z] = (0, 0, 0)

    return image
