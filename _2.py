import cv2

mavi = [255, 0, 0]
wan = cv2.imread("wan.jpeg")

replicate = cv2.copyMakeBorder(wan, 10, 10, 10, 10, cv2.BORDER_REPLICATE)

constant = cv2.copyMakeBorder(wan, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=mavi)
# düz mavi renkli border yapar

#cv2.imshow("original", wan)
#cv2.imshow("replicate", replicate)
#cv2.imshow("constant", constant)

#cv2.waitKey(0)

roi = wan[30:120, 100:200]
# we choose our roi, then we will plot a rectangle to there
cv2.rectangle(wan, (30, 120), (100, 200), (0, 255, 255), 2)
# dikdörtgeni çizdik

cv2.imshow("roi", wan)
cv2.waitKey(0)
cv2.destroyAllWindows()


