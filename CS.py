import cv2
import numpy as np

print (cv2.__version__)
print (np.__version__)

origimg = cv2.imread("example2.jpg")
"""
# font
font = cv2.FONT_HERSHEY_SIMPLEX

# org
org = (50, 50)

# fontScale
fontScale = 1

# Blue color in BGR
color = (128, 0, 128)

# Line thickness of 2 px
thickness = 2

# Using cv2.putText() method
image = cv2.putText(origimg, 'OpenCV', org, font,
                   fontScale, color, thickness, cv2.LINE_AA)
"""
cv2.imshow("orig", origimg)
#print ("orig:\n {0}".format(origimg))

cv2.waitKey(0)
cv2.destroyAllWindows()

## Получаем HSV-представление для цвета

red = np.uint8([[[0, 0, 255]]])
green = np.uint8([[[0, 255, 0]]])
deep_blue = np.uint8([[[255, 0, 0]]])
orange = np.uint8([[[0, 128, 255]]])
yellow = np.uint8([[[0, 255, 255]]])
blue = np.uint8([[[255, 255, 0]]])
purple = np.uint8([[[255, 0, 255]]])
violet = np.uint8([[[128, 0, 128]]])

scolor = orange
scolor_hsv = cv2.cvtColor(scolor,cv2.COLOR_BGR2HSV)
print(scolor_hsv)

size = 1, 1, 3
m = np.zeros(size, dtype=np.uint8)
m[:] = scolor_hsv

scolor_hue, scolor_saturation, scolor_value = m[0][0]###
print(scolor_hue)

hue_size = 20

scolor_hue_min = scolor_hue - hue_size
if scolor_hue_min < 0:
    scolor_hue_min = 0

scolor_hue_max = scolor_hue + hue_size
if scolor_hue_max > 360:
    scolor_hue_max = 360

hsvimg = cv2.cvtColor(origimg, cv2.COLOR_BGR2HSV)
cv2.imshow("hsv1", hsvimg)
cv2.waitKey(0)

#hsv_min = np.array((53, 0, 0), np.uint8)
scolor_low = np.array((scolor_hue_min, 0, 0), np.uint8)
scolor_high = np.array((scolor_hue_max, 255, 255), np.uint8)
curr_mask = cv2.inRange(hsvimg, scolor_low, scolor_high)
cv2.imshow ("cm", curr_mask)
cv2.waitKey(0)


rows,cols = curr_mask.shape
#print(rows)
#print(cols)

for i in range(rows):
    for j in range(cols):
        if i==0 or i==rows-1 or j==0 or j==cols-1:
            curr_mask[i,j] = 0
            #print(curr_mask[i,j])
            #print ("curr_mask:\n {0}".format(curr_mask[i,j]))

cv2.imshow ("cm1", curr_mask)
cv2.waitKey(0)

gb_curr_mask = cv2.GaussianBlur(curr_mask,(9,9),cv2.BORDER_DEFAULT)
cv2.imshow ("gbcm", gb_curr_mask)
cv2.waitKey(0)
for i in range(rows):
    for j in range(cols):
        if gb_curr_mask[i,j] < 140:
            gb_curr_mask[i,j] = 0
        else:
            gb_curr_mask[i,j] = 255
cv2.imshow ("gbcm1", gb_curr_mask)
cv2.waitKey(0)

"""newmask = np.zeros((rows,cols,1), np.uint8)
for i in range(rows):
    for j in range(cols):
        if curr_mask[i,j] == 255:
            newmask[i,j] = [255]

contours, hierarchy = cv2.findContours(newmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
for pic, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    #perimeter = cv2.arcLength(contour, True)
    #print(perimeter)
    if area < 600:
        cv2.drawContours(newmask, contour, -1, [0, 0, 0], -1)
        #print(perimeter)

contours, hierarchy = cv2.findContours(newmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
for pic, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    #perimeter = cv2.arcLength(contour, True)
    if area < 600:
        cv2.drawContours(newmask, contour, -1, [255, 255, 255], -1)
        #print(perimeter)

cv2.imshow ("nm", newmask)
cv2.waitKey(0)"""

hsvimg[gb_curr_mask > 0] = ([scolor_hue,255,255])#curr_mask
origimg[gb_curr_mask == 0] = ([0,0,0])
cv2.imshow("hsv2", hsvimg)
cv2.waitKey(0)
cv2.imshow ("orig2", origimg)
cv2.waitKey(0)
cv2.imwrite("orig2_orange.jpeg", origimg)###############################################################################################################################

"""for i in range(rows):
    for j in range(cols):
        if hsvimg[i,j] is not [scolor_hue,255,255]:
            origimg[i,j] = ([0,0,0])
cv2.imshow ("orig111", origimg)
cv2.waitKey(0)"""

## Преобразование HSV-изображения к оттенкам серого для дальнейшего
## оконтуривания
RGB_again = cv2.cvtColor(hsvimg, cv2.COLOR_HSV2RGB)
gray = cv2.cvtColor(RGB_again, cv2.COLOR_RGB2GRAY)
cv2.imshow ("gray", gray)
cv2.waitKey(0)
ret, threshold = cv2.threshold(gray, 90, 255, 0)
cv2.imshow ("threshold", threshold)
cv2.waitKey(0)
contours, hierarchy = cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(origimg, contours, -1, (0, 0, 255), 3)
cv2.imshow ("origc", origimg)
cv2.waitKey(0)
#origimg[hsvimg != ([scolor_hue,255,255])] = (0, 0, 0)
#print ("hsvimg:\n {0}".format(hsvimg[0,0]))


#print(curr_mask[0,0])
# | i==rows-1 | j==0 | j==cols-1

#print ("curr_mask:\n {0}".format(curr_mask))
#, dtype=cv2.CV_8UC3, order='C'
"""bgr_again = np.zeros((rows, cols), dtype=[cv2.CV_8UC3], order='C')
for i in range(rows):
    for j in range(cols):
        if curr_mask[i,j] == 0:
            bgr_again[i,j] = [0, 0, 0]
        elif curr_mask[i,j] == 255:
            bgr_again[i,j] = [255, 255, 255]

#cv2.imshow ("bgr_again",bgr_again)
#cv2.waitKey(0)

contours, hierarchy = cv2.findContours(bgr_again, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
for pic, contour in enumerate(contours):
    perimeter = cv2.arcLength(contour, True)
    #print(perimeter)
    if perimeter < 400:
        cv2.drawContours(bgr_again, contour, 0, [0, 0, 0], -1)
        #print(perimeter)

contours, hierarchy = cv2.findContours(bgr_again, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
for pic, contour in enumerate(contours):
    perimeter = cv2.arcLength(contour, True)
    if perimeter < 400:
        cv2.drawContours(bgr_again, contour, 0, [255, 255, 255], -1)
        #print(perimeter)

cv2.imshow ("bgr_again",bgr_again)
cv2.waitKey(0)"""
#bigger = max(contours, key=lambda item: cv2.contourArea(item))
#the_mask = np.zeros_like(curr_mask)
#cv2.drawContours(the_mask, [bigger], -1, (255, 255, 255), cv2.FILLED)
#res = cv2.bitwise_and(curr_mask, curr_mask, mask = the_mask)
#cv2.imshow ("res", res)
#cv2.waitKey(0)
#hsvimg[curr_mask > 0] = ([scolor_hsv, 255, 255]) #([75,255,200])
#cv2.imshow("hsv2", hsvimg)

#cv2.waitKey(0)
cv2.destroyAllWindows()
