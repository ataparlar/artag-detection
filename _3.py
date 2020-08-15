import cv2

wan = cv2.imread("wan.jpeg")
opencv = cv2.imread("opencv.jpg")


add1 = cv2.add(wan, opencv)
# 2 resmi ağırlıkları eşit olarak topla

add2 = cv2.addWeighted(wan, 0.4, opencv, 0.6, 0)
# ağırlıkları kendimiz vereceğimiz şekilde toplayabiliyoruz.
# en sondaki gamma parametresi

