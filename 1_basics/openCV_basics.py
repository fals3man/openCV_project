import cv2
import numpy as np
from matplotlib import pyplot as plt

input=cv2.imread(r"...\...\...\image.jpg")

#comment our the required parts


#GRAYSCALE
# gray_image=cv2.cvtColor(input,cv2.COLOR_BGR2GRAY)
# row,col=1,2
# fig,axs=plt.subplots(row,col,figsize=(15,10))
# fig.tight_layout()
# axs[0].imshow(cv2.cvtColor(input,cv2.COLOR_BGR2RGB))
# axs[0].set_title("Original")
# axs[1].imshow(cv2.cvtColor(gray_image,cv2.COLOR_BGR2RGB))
# axs[1].set_title("Gray Image")
# plt.show()
# cv2.imwrite("gray_image.png",gray_image)


#HSV
# hsv_image=cv2.cvtColor(input,cv2.COLOR_BGR2HSV)
# row,col=2,2
# fig,axs=plt.subplots(row,col,figsize=(15,8))
# fig.tight_layout()
# axs[0][0].imshow(cv2.cvtColor(hsv_image,cv2.COLOR_BGR2RGB))
# axs[0][0].set_title("HSV")
# axs[0][1].imshow(cv2.cvtColor(hsv_image[:,:,0],cv2.COLOR_BGR2RGB))
# axs[0][1].set_title("Hue channel")
# axs[1][0].imshow(cv2.cvtColor(hsv_image[:,:,1],cv2.COLOR_BGR2RGB))
# axs[1][0].set_title("Saturation channel")
# axs[1][1].imshow(cv2.cvtColor(hsv_image[:,:,2],cv2.COLOR_BGR2RGB))
# axs[1][1].set_title("value channel")
# plt.show()


#MERGE 
# B,G,R=cv2.split(input)
# row,col=1,3
# fig,axs=plt.subplots(row,col,figsize=(15,10))
# fig.tight_layout()
# axs[0].imshow(cv2.cvtColor(R,cv2.COLOR_BGR2RGB))
# axs[0].set_title("Red")
# axs[1].imshow(cv2.cvtColor(G,cv2.COLOR_BGR2RGB))
# axs[1].set_title("Green")
# axs[2].imshow(cv2.cvtColor(B,cv2.COLOR_BGR2RGB))
# axs[2].set_title("Blue")
# plt.show()
# row,col=1,2
# fig,axs=plt.subplots(row,col,figsize=(15,10))
# fig.tight_layout()
# merged=cv2.merge([B,G,R])
# axs[0].imshow(cv2.cvtColor(merged,cv2.COLOR_BGR2RGB))
# axs[0].set_title("Merged")
# merged=cv2.merge([B,G,R+100])
# axs[1].imshow(cv2.cvtColor(merged,cv2.COLOR_BGR2RGB))
# axs[1].set_title("Merged with red amplified")
# plt.show()
# cv2.imwrite("merged_with_red_amplified.png",merged)


#RGB aka BGR
# B,G,R=cv2.split(input)
# zeros=np.zeros(input.shape[:2],dtype="uint8")
# row,col=1,3
# fig,axs=plt.subplots(row,col,figsize=(15,10))
# fig.tight_layout()
# axs[0].imshow(cv2.merge([R,zeros,zeros]))
# axs[0].set_title("Red")
# axs[1].imshow(cv2.merge([zeros,G,zeros]))
# axs[1].set_title("Green")
# axs[2].imshow(cv2.merge([zeros,zeros,B]))
# axs[2].set_title("Blue")
# plt.show()
# cv2.imwrite("Blue.png",cv2.merge([R,zeros,zeros]))
# cv2.imwrite("Green.png",cv2.merge([zeros,G,zeros]))
# cv2.imwrite("Red.png",cv2.merge([zeros,zeros,B]))


#HISTOGRAMS
# hist=cv2.calcHist([input],[0],None,[256],[0,256])
# plt.hist(input.ravel(),256,[0,256])
# plt.show()
# cv2.imwrite("Histogram.png",(hist))
# color=('b','g','r')
# for i,col in enumerate(color):
#     hist1=cv2.calcHist([input],[i],None,[256],[0,256])
#     plt.plot(hist1,color=col)
#     plt.xlim([0,256])
# plt.show()    
# row,col=1,2
# fig,axs=plt.subplots(row,col,figsize=(15,10))
# fig.tight_layout()
# axs[0].imshow((input))
# axs[0].set_title("Matplotlib colors")
# axs[1].imshow(cv2.cvtColor(input,cv2.COLOR_BGR2RGB))
# axs[1].set_title("Original RGB colors")
# plt.show()
