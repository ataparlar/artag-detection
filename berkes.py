import cv2
import numpy as np
import json


"""
KODA BAKMADAN ÖNCE
Yeşil ile gösterilenler adaylardır 
Mavi ile gösterilenler ise onaylanmış adaylardır(markers)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
Bu kod 5x5 lik kodları algılamak için tasarlanmıştır tags isimli sınıfta "valids" isimli listede
kabul edilen her artag'in iç kodları tanımlanmışır. İsterseniz yeni kodlar ekleyebilirsiniz.
*PARAMETRELER
params klasörü hemen hemen bütün parametrelerin depolandığı yerdir
-threshConstant: eşiklemenin hassasiyeti ile oynar ne kadar düşük o kadar hassaslaşır
-threshWinSize: parametleri ile oynanabilir ama tek sayı olmasına dikkat ediniz
-minAreaRate: belirlenen adayların sahip olması gereken en küçük alanın hesaplanmasında kullanılır
              bunu azaltmanız daha küçük adaylar görmenizi sağlar
-maxAreaRate: minAreaRate parametresine benzer ama ters etkiye neden olur
-resizeRate: bu parammetrenin artırılması belli bir noktaya kadar algoritma kesinliğini arttırır lakin 
             işlem gücü harcamasına neden olur
-cellMarginRate: her bite bakarken onun tamamına değil de biraz daha içine bakarız. Bu parametre her hücrenin 
                 yüzde kaç içine bakmamız gerekiğini belirler
-markerSizeInBits & borderSizeInBits: Artag'in sınırlarının ve iç kodunun bit boyutu yazılmıştır lakin bunun
                                      değiştirilmesi tavsiye edinilmez çünkü tanımlı bütün artaglar bu iki 
                                      parametreye göre tanımlı.
-configFileName: kamera kalibrasyon dosyasının bulunduğu konum(eğer aynı klasörde ise ismi yetiyor
-undistortImg: kamera kalibrasyonu hala deneme aşamasında oluğu için isterseniz undistortion işlemini engelleyebilirsiniz
-showCandidate showMarkers: showTresholded bu üçü isminden de anlaşıacağı üzere istediğiniz şeyleri kapatıp açabilirsiniz
"""


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


"""def load_camera_params(filename='default.json'):
    with open(filename, 'r') as loadFile:
        data = json.load(loadFile)
        mtx = np.array(data['mtx'])
        dist = np.array(data['dist'])
    return mtx, dist"""


# çok ağır çalışıyor + tam çalışmıyor (bu fonksiyondan vazgeçilebilir)
def remove_close_candidates(candidates):
    newCandidates = list()

    for i in range(len(candidates)):
        for j in range(len(candidates)):
            # adayımızın kendisini kontrol etmesini istemeyiz
            if i == j:
                continue

            minPerimeter = min(cv2.arcLength(candidates[i], True), cv2.arcLength(candidates[j], True))

            # fc ilk köşe
            for fc in range(4):
                disSq = 0
                for c in range(4):
                    modC = (fc + c) % 4
                    dx = candidates[j][c][0][0] - candidates[i][modC][0][0]
                    dy = candidates[j][c][0][1] - candidates[i][modC][0][1]
                    disSq += dx * dx + dy * dy
                disSq /= 4

                minDisPixels = minPerimeter * params.minMarkerDisRate

                if disSq < minDisPixels * minDisPixels:
                    if cv2.contourArea(candidates[i]) > cv2.contourArea(candidates[j]):
                        newCandidates.append(candidates[i])
                    else:
                        newCandidates.append(candidates[j])

    # eğer newCandidates boş ise zaten herhangi bir filtreleme
    # olmamıştır bu yüzden eskisini döndeririz
    if len(newCandidates):
        return newCandidates
    else:
        return candidates


def has_close_corners(candidate):
    minDisSq = float("inf")

    for i in range(len(candidate)):
        dx = candidate[i][0][0] - candidate[(i+1)%4][0][0]
        dy = candidate[i][0][1] - candidate[(i+1)%4][0][1]
        dsq = dx * dx + dy * dy
        minDisSq = min(minDisSq, dsq)

    minDisPixel = candidate.size * params.minCornerDisRate
    if minDisSq < minDisPixel * minDisPixel:
        return True
    else:
        return False


def sort_corners(corners):
   dx1 = corners[1][0] - corners[0][0]
   dy1 = corners[1][1] - corners[0][1]
   dx2 = corners[2][0] - corners[0][0]
   dy2 = corners[2][1] - corners[0][1]

   crossproduct = (dx1 * dy2) - (dy1 * dx2)

   if crossproduct > 0:
       corners[1], corners[3] = corners[3], corners[1]


def get_corners(candidate):
    corners = np.array([
        [candidate[0][0][0], candidate[0][0][1]],
        [candidate[1][0][0], candidate[1][0][1]],
        [candidate[2][0][0], candidate[2][0][1]],
        [candidate[3][0][0], candidate[3][0][1]]
    ], dtype="float32")
    return corners


def get_candate_img(candidate, frame):
    corners = get_corners(candidate)
    sort_corners(corners)

    (tl, tr, br, bl) = corners
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array(
        [[0, 0],
         [maxWidth - 1, 0],
         [maxWidth-1, maxHeight-1],
         [0, maxHeight - 1]
         ], dtype="float32"
    )

    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(frame, M, (maxWidth, maxHeight), borderMode=cv2.INTER_NEAREST)

    return warped, maxWidth, maxHeight


def validate_candidates(candidates, frame):
    markers = list()
    for can in candidates:
        candidate_img, maxWidth, maxHeight = get_candate_img(can, frame)
        ret, candidate_img = cv2.threshold(candidate_img, 125, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        ust = candidate_img[0: int(round(maxHeight/6)), 0:]
        sol = candidate_img[0:, 0: int(round(maxWidth/6))]
        alt = candidate_img[int(round((maxHeight/6)*5)): maxWidth, 0:]
        sag = candidate_img[0:, int(round((maxWidth/6)*5)):]
        black_list = [ust, sol, alt, sag]
        cv2.imshow("alt", alt)
        for edge in black_list:
            bits = extract_bits(edge)

            print(bits)
            print("------------------------")
            if np.count_nonzero(bits) > 3:
                #print("*********************")
                #index = np.argwhere(np.count_nonzero(bits) > 3)
                #np.delete(candidates, index)
                candidates.remove(can)
                print(can)
                break
        # burada AR tag'in kenarlarının %20'sinin siyah olup olmadığını kontrol eden bir if bloğu olmalı

        bits = extract_bits(candidate_img)
        bits = np.transpose(bits)

        for valid in tags.valids:
            validMarker = valid.copy()
            for i in range(4):
                if np.array_equal(bits, validMarker):
                    markers.append(can)
                    break
                validMarker = np.rot90(validMarker)

        #bitimg = recreate_img(bits)
        #cv2.imshow("bits", bitimg)
        #cv2.imshow("otsu", candidate_img)
        cv2.imshow("cand", candidate_img)
    return markers


# extract_bits fonksiyonu çalışmasını kontrol etmek için oluşturuldu
def recreate_img(bits):
    cellSize = 30
    img = np.zeros((bits.shape[0] * cellSize, bits.shape[1] * cellSize, 1))

    for j in range(bits.shape[0]):
        for i in range(bits.shape[1]):
            if bits[j, i] == 0:
                continue
            for x in range(cellSize):
                ix = i * cellSize + x
                for y in range(cellSize):
                    iy = j * cellSize + y
                    img[iy, ix] = 255

    return img


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


def detect_candidates(grayImg):
    th = cv2.adaptiveThreshold(grayImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, params.threshConstant)
    cnts = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[-2]

    if params.showTresholded is True:
        cv2.imshow('treshold', th)

    # ayıklama
    candidates = list()
    for c in cnts:
        # boyut kontrolü
        maxSize = int(max(gray.shape) * params.maxAreaRate)
        minSize = int(max(gray.shape) * params.minAreaRate)
        if c.size > maxSize or c.size < minSize:
            continue

        approxCurve = cv2.approxPolyDP(c, len(c) * params.accuracyRate, True)
        # karesellik kontrolü
        if len(approxCurve) is not 4 or cv2.isContourConvex(approxCurve) is False:
            continue

        # köşler birbirlerine çokmu yakın ona bakılır
        if has_close_corners(approxCurve):
            continue

        # testleri geçerse ekle
        candidates.append(approxCurve)

    return candidates


def find_center(marker):
    (x, y), r = cv2.minEnclosingCircle(marker)
    return int(x), int(y)


# ANA ALGORİTMA BAŞLANGICI
camera = cv2.VideoCapture(0)
#mtx, dist = load_camera_params(filename=params.configFileName)

while True:
    _, frame = camera.read()
    frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # kamera kalibrasyonu hala deneme aşamasında
    """if params.undistortImg is True:
        frame = cv2.undistort(frame, mtx, dist)"""

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    candidates = detect_candidates(gray)

    if len(candidates) > 0:
        #candidates = remove_close_candidates(candidates)
        markers = validate_candidates(candidates, gray)
        print(markers)

        if params.showCandidate is True:
            cv2.drawContours(frame, candidates, -1, (0, 255, 0), 2)

        if len(markers) > 0 and params.showMarkers is True:
            cv2.drawContours(frame, markers, -1, (255, 0, 0), 3)

    cv2.imshow('frame', frame)
    if cv2.waitKey(10) == 27:  # esc ile çıkar
        break