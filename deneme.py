import cv2

video = cv2.VideoCapture(0)

while True:
    ret, cam = video.read()

    cv2.imshow("cam", cam)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()

