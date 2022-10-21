#...................................................................
# #LIVE SKETCH USING WEBCAM
#...................................................................

import cv2
import numpy as np
def sketch(img):
    img_gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img_blur=cv2.GaussianBlur(img_gray,(5,5),0)
    edges=cv2.Canny(img_blur,10,70)
    ret,mask=cv2.threshold(edges,70,255,cv2.THRESH_BINARY)
    return mask
cap=cv2.VideoCapture(0)
while True:
    ret,frm=cap.read()
    cv2.imshow('Live sketcher',sketch(frm))
    if(cv2.waitKey(1)==13):      #13 is Enter Key
        break
cap.release()
cv2.destroyAllWindows()  
# cv.destroyallwindows
