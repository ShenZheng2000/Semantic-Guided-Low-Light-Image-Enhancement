# The purpose is to cut images acc. to specific height and width
import cv2


#n = 15
h = 540
w = 960

index = 1224

'''
img = cv2.imread("F1.png")
cropped = img[0:h , 0:w ]  # 裁剪坐标为[y0:y1, x0:x1]
cv2.imwrite("F1Crop.png", cropped)

img = cv2.imread("./Clean_Images/low/" + str(index) + ".png")
print(img.shape)
cropped = img[0:h , 0:w ]  # 裁剪坐标为[y0:y1, x0:x1]
cv2.imwrite("./Clean_Images/highCUT/" + str(index) + ".png", cropped)

'''


for i in range(0, index + 1):

    img = cv2.imread("data/" + "frame" + str(i) +".jpg")
    print(img.shape)
    cropped = img[0:h, 0:w]  # 裁剪坐标为[y0:y1, x0:x1]

    cv2.imwrite("dataCUT/" + "frame" + str(i) +".jpg", cropped)


