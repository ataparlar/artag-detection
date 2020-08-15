import cv2
import numpy as np

class tags:
    valids = [
        np.array([
            [1, 1, 0, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 0, 1, 0, 1],
            [1, 0, 1, 1, 0],
            [1, 0, 1, 1, 0],
        ], dtype=int),
        np.array([
            [1, 1, 0, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 0, 1, 0, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ], dtype=int),
        np.array([
            [1, 1, 0, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 0, 1, 0, 1],
            [0, 0, 1, 1, 0],
            [1, 1, 1, 0, 1],
        ], dtype=int),
        np.array([
            [1, 1, 0, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 0, 1, 0, 1],
            [0, 1, 1, 1, 1],
            [1, 0, 1, 0, 0],
        ], dtype=int),
        np.array([
            [1, 1, 0, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 0, 1, 0, 1],
            [1, 0, 1, 1, 1],
            [0, 1, 1, 0, 0],
        ], dtype=int),
        np.array([
            [1, 1, 0, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 0, 1, 0, 1],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
        ], dtype=int),
        np.array([
            [1, 1, 0, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 0, 1, 0, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1],
        ], dtype=int),
        np.array([
            [1, 1, 0, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 0, 1, 0, 1],
            [1, 1, 1, 1, 0],
            [0, 0, 1, 0, 1],
        ], dtype=int),
        np.array([
            [1, 1, 0, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 0, 1, 0, 1],
            [0, 0, 1, 0, 1],
            [1, 1, 1, 1, 0],
        ], dtype=int),
    ]


class params:
    threshConstant = 3
    threshWinSizeMax = 23
    threshWinSizeMin = 3
    threshWinSizeStep = 10
    accuracyRate = 0.02
    minAreaRate = 0.03
    maxAreaRate = 6
    minCornerDisRate = 2.5
    minMarkerDisRate = 1
    resizeRate = 4
    cellMarginRate = 0.13
    markerSizeInBits = 5
    borderSizeInBits = 2
    configFileName = 'logi-g922-config.json'
    undistortImg = True
    showCandidate = True
    showMarkers = True
    showTresholded = True

def resize_roi(roi, scale_percent):
    width = int(roi.shape[1] * scale_percent / 100)
    height = int(roi.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized_roi = cv2.resize(roi, dim, interpolation=cv2.INTER_AREA)
    return resized_roi


def find_small_candidate(approx):
    approx_x_list, approx_y_list = np.array([]), np.array([])
    approx_x_list, approx_y_list = approx_x_list.astype('int32'), approx_y_list.astype('int32')
    for point in approx:  # point, approx listesi içindeki katman. poligon'un noktalarını temsil ediyor.
        approx_x_list = np.append(approx_x_list, point[0, 0])
        approx_y_list = np.append(approx_y_list, point[0, 1])

    cv2.putText(cam,
                str(len(approx)),  # eklenecek yazı
                (point[0, 0] - 10, point[0, 1] - 10),  # nereye eklenecek
                cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255)  # font tipi, büyüklüğü ve rengi
                )
    roi = cam[np.amin(approx_y_list)-10:np.amax(approx_y_list)+10, np.amin(approx_x_list)-10:np.amax(approx_x_list)+10]
    #resized_roi = resize_roi(roi, 200)

    #erosion = cv2.erode(roi, kernel, iterations=1)
    #dilation = cv2.dilate(roi, kernel, iterations=1)

    #small_blur = cv2.GaussianBlur(resized_roi, (7, 7), 0)

    #filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    #sharpened = cv2.filter2D(small_blur, -1, filter)

    if 7 > len(approx) > 4:
        cv2.drawContours(roi, [approx], 0, (255, 255, 255), 7)


def get_corners(candidate):
    corners = np.array([
        [candidate[0][0][0], candidate[0][0][1]],
        [candidate[1][0][0], candidate[1][0][1]],
        [candidate[2][0][0], candidate[2][0][1]],
        [candidate[3][0][0], candidate[3][0][1]]
    ], dtype="float32")
    return corners


# ------------------------------------------
# ------------------------------------------
# ------------------------------------------


def sort_corners(corners):
   dx1 = corners[1][0] - corners[0][0]
   dy1 = corners[1][1] - corners[0][1]
   dx2 = corners[2][0] - corners[0][0]
   dy2 = corners[2][1] - corners[0][1]

   crossproduct = (dx1 * dy2) - (dy1 * dx2)

   if crossproduct > 0:
       corners[1], corners[3] = corners[3], corners[1]


def get_candidate_image(candidate, frame):
    corners = get_corners(candidate)
    sort_corners(corners)

    (tl, tr, br, bl) = corners
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))  #

    dst = np.array(
        [[0, 0],
         [maxWidth - 1, 0],
         [maxWidth - 1, maxHeight - 1],
         [0, maxHeight - 1]
         ], dtype="float32"
    )

    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(frame, M, (maxWidth, maxHeight), borderMode=cv2.INTER_NEAREST)
    cv2.imshow("warped", warped)
    return warped


def validate_candidates(candidates, gray):
    markers = list()
    for candidate in candidates:
        candidate_image = get_candidate_image(candidate, gray)
        ret, candidate_image = cv2.threshold(candidate_image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        bits = extract_bits(candidate_image)
        bits = np.transpose(bits)

        for valid in tags.valids:
            validMarker = valid.copy()
            for i in range(4):
                if np.array_equal(bits, validMarker):
                    markers.append(candidate)
                    break
                validMarker = np.rot90(validMarker)

        return markers


def resize_img(inputImg):
    w = int(inputImg.shape[1]*params.resizeRate)
    h = int(inputImg.shape[0]*params.resizeRate)
    outputImg = cv2.resize(inputImg, (w,h))
    return outputImg


def extract_bits(img):
    img = resize_img(img)

    # artag boyutları şimdilik fonksiyon içinde tanımlı
    # ileride belli bir parametreye bağlanacak
    markerSize = params.markerSizeInBits
    borderSize = params.borderSizeInBits

    markerSizeWithBorders = markerSize + 2 * borderSize
    bitmap = np.zeros((markerSize, markerSize), dtype=int)
    cellWidth = int(img.shape[1] / markerSizeWithBorders)
    cellHeight = int(img.shape[0] / markerSizeWithBorders)

    inner_rg = img[borderSize*cellHeight:(markerSizeWithBorders-borderSize)*cellHeight,
               borderSize*cellWidth:(markerSizeWithBorders-borderSize)*cellWidth]

    marginX = int(cellWidth * params.cellMarginRate)
    marginY = int(cellHeight * params.cellMarginRate)

    # her bit için
    for j in range(markerSize):
        Ystart = j * cellHeight
        for i in range(markerSize):
            Xstart = i * cellWidth
            bitImg = inner_rg[Ystart+marginY:Ystart+cellHeight-marginY, Xstart+marginX:Xstart+cellWidth-marginX]
            if np.count_nonzero(bitImg) / bitImg.size > 0.5:
                bitmap[j][i] = 1

    return bitmap




def has_close_corners(candidate):
    minDisSq = float("inf")

    for i in range(len(candidate)):
        dx = candidate[i][0][0] - candidate[(i+1)%4][0][0]
        dy = candidate[i][0][1] - candidate[(i+1)%4][0][1]
        dsq = dx * dx + dy * dy
        minDisSq = min(minDisSq, dsq)

    minDisPixel = candidate.size * 2.5
    if minDisSq < minDisPixel * minDisPixel:
        return True
    else:
        return False




video = cv2.VideoCapture(0)

while True:
    ret, cam = video.read()
    blur = cv2.GaussianBlur(cam, (5, 5), 0)
    camgray0 = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    # kernel = np.ones((3, 3), np.uint8)
    # erosion = cv2.erode(camgray0, kernel, iterations=2)
    # dilation = cv2.dilate(camgray0, kernel, iterations=2)
    # gürültü azaltmak için normalde ama çizgiler daha kalınlaştı

    threshold = cv2.adaptiveThreshold(camgray0, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 3)

    # ret, threshold = cv2.threshold(camgray, 127, 255, cv2.THRESH_BINARY)
    contours = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]

    candidates = list()

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.015*cv2.arcLength(contour, True), True)

        """if 9 > len(approx) > 4 and cv2.contourArea(contour) > 20:
            cv2.drawContours(cam, [approx], 0, (0, 0, 255), 3)
            find_small_candidate(approx)"""
        if len(approx) != 4:
            continue

        if has_close_corners(approx):
            continue
            # cv2.drawContours(cam, [approx], 0, (0, 255, 0), 3)

        candidates.append(approx)

    if len(candidates) > 0:
        markers = validate_candidates(candidates, camgray0)
        print(markers)

        if params.showCandidate is True:
            cv2.drawContours(cam, candidates, -1, (0, 255, 0), 2)

        if len(markers) > 0 and params.showMarkers is True:
            cv2.drawContours(cam, markers, -1, (255, 0, 0), 3)

    # print(candidates)

    cv2.imshow("cam", cam)
    cv2.imshow("threshold", threshold)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()

