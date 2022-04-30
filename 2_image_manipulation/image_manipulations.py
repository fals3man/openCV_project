import cv2
from cv2 import add
import numpy as np
from matplotlib import pyplot as plt
image=cv2.imread(r'C:\Users\sraka\Downloads\mrrobot.jpg')
h,w=image.shape[:2]

#.......................................................................................................................................................
#COMMENT OUT REQUIRED PART
#.......................................................................................................................................................


#.......................................................................................................................................................
#IMAGE TRANSLATION
#.......................................................................................................................................................
# quarter_height,quarter_weight=h/4,w/4
# T=np.float32([[1,0,quarter_height],[0,1,quarter_weight]])    #translation matrix
# print (T)
# image_translation=cv2.warpAffine(image,T,(w,h))
# plt.imshow(cv2.cvtColor(image_translation,cv2.COLOR_BGR2RGB))
# plt.title('Translated Image')
# plt.show()

#.......................................................................................................................................................
# #IMAGE ROTATION
#.......................................................................................................................................................
# #Method1
# rotat_matrix=cv2.getRotationMatrix2D((w/2,h/2),90,0.5)
# rotat_img=cv2.warpAffine(image,rotat_matrix,(w,h))
# plt.imshow(cv2.cvtColor(rotat_img,cv2.COLOR_BGR2RGB))
# plt.title('Rotated Image')
# plt.show()
# #Method2
# rotat_img1=cv2.transpose(image)
# plt.imshow(cv2.cvtColor(rotat_img1,cv2.COLOR_BGR2RGB))
# plt.title('Rotated Image M2')
# plt.show()

#.......................................................................................................................................................
# #RESIZING,INTERPOLATION
#.......................................................................................................................................................
# row,col=1,3
# fig,axs=plt.subplots(row,col,figsize=(15,10))
# fig.tight_layout()
# scaled_img=cv2.resize(image,None,fx=0.25,fy=0.25)
# axs[0].imshow(cv2.cvtColor(scaled_img,cv2.COLOR_BGR2RGB))
# axs[0].set_title('Scaling-Linear Interpolation')
# scaled_img1=cv2.resize(image,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
# axs[1].imshow(cv2.cvtColor(scaled_img1,cv2.COLOR_BGR2RGB))
# axs[1].set_title('Scaling-Cubic Interpolation')
# scaled_img2=cv2.resize(image,(690,420),interpolation=cv2.INTER_AREA)
# axs[2].imshow(cv2.cvtColor(scaled_img2,cv2.COLOR_BGR2RGB))
# axs[2].set_title('Scaling-Skewed size')
# plt.show()

#.......................................................................................................................................................
# #IMAGE PYRAMIDS
#.......................................................................................................................................................
# row,col=1,3
# fig,axs=plt.subplots(row,col,figsize=(15,10))
# fig.tight_layout()
# smaller_img=cv2.pyrDown(image)
# larger_img=cv2.pyrUp(image)
# axs[0].imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
# axs[0].set_title('Original Image')
# cv2.imwrite("Original_image.png",image)
# axs[1].imshow(cv2.cvtColor(smaller_img,cv2.COLOR_BGR2RGB))
# axs[1].set_title('Smaller Image')
# cv2.imwrite("Smaller_image.png",smaller_img)
# axs[2].imshow(cv2.cvtColor(larger_img,cv2.COLOR_BGR2RGB))
# axs[2].set_title('Larger Image')
# cv2.imwrite("Larger_image.png",larger_img)
# plt.show()

#.......................................................................................................................................................
# #CROPPING
#.......................................................................................................................................................
# h,w=image.shape[:2]
# start_row,start_col=int(h*.25),int(w*0.25)
# end_row,end_col=int(h*.75),int(w*0.75)
# cropped_img=image[start_row:end_row,start_col:end_col]
# row,col=1,2
# fig,axs=plt.subplots(row,col,figsize=(15,10))
# fig.tight_layout()
# axs[0].imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
# axs[0].set_title('Original Image')
# axs[1].imshow(cv2.cvtColor(cropped_img,cv2.COLOR_BGR2RGB))
# axs[1].set_title('Cropped Image')
# cv2.imwrite("Cropped_image.png",cropped_img)
# plt.show()

#.......................................................................................................................................................
# #ARITHMETIC OPERATIONS
#.......................................................................................................................................................
# M=np.ones(image.shape,dtype="uint8")*200
# row,col=1,3
# fig,axs=plt.subplots(row,col,figsize=(15,10))
# fig.tight_layout()
# axs[0].imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
# axs[0].set_title('Original Image')
# added_img=cv2.add(image,M)
# axs[1].imshow(cv2.cvtColor(added_img,cv2.COLOR_BGR2RGB))
# axs[1].set_title('Added Image')
# cv2.imwrite("Added_image.png",added_img)
# subtracted_image=cv2.subtract(image,M)
# axs[2].imshow(cv2.cvtColor(subtracted_image,cv2.COLOR_BGR2RGB))
# axs[2].set_title('Subtracted Image')
# cv2.imwrite("Subtracted_image.png",subtracted_image)
# plt.show()

#.......................................................................................................................................................
# #BLURRING AND CONVOLUTIONS
#.......................................................................................................................................................
# row,col=1,2
# fig,axs=plt.subplots(row,col,figsize=(15,10))
# fig.tight_layout()
# kernel_3=np.ones((3,3),np.float32)/9
# blurred=cv2.filter2D(image,-1,kernel_3)
# axs[0].imshow(cv2.cvtColor(blurred,cv2.COLOR_BGR2RGB))
# axs[0].set_title('3x3 kernel blurring')
# kernel_7=np.ones((7,7),np.float32)/49
# blurred1=cv2.filter2D(image,-1,kernel_7)
# axs[1].imshow(cv2.cvtColor(blurred1,cv2.COLOR_BGR2RGB))
# axs[1].set_title('7x7 kernel blurring')
# plt.show()
# row,col=2,2
# fig,axs=plt.subplots(row,col,figsize=(10,5))
# fig.tight_layout()
# blur=cv2.blur(image,(7,7))
# axs[0][0].imshow(cv2.cvtColor(blur,cv2.COLOR_BGR2RGB))
# axs[0][0].set_title('Average Blur')
# cv2.imwrite('Average_blur.png',blur)
# axs[0][0].imshow(cv2.cvtColor(blur,cv2.COLOR_BGR2RGB))
# axs[0][0].set_title('Average Blur')
# cv2.imwrite('Average_blur.png',blur)
# Gaussian=cv2.GaussianBlur(image,(7,7),0)
# axs[0][1].imshow(cv2.cvtColor(Gaussian,cv2.COLOR_BGR2RGB))
# axs[0][1].set_title('Gaussian Blur')
# cv2.imwrite('Gaussian_blur.png',Gaussian)
# median=cv2.medianBlur(image,5)
# axs[1][0].imshow(cv2.cvtColor(median,cv2.COLOR_BGR2RGB))
# axs[1][0].set_title('Median Blur')
# cv2.imwrite('Median_blur.png',median)
# bilateral=cv2.bilateralFilter(image,9,75,75)
# axs[1][1].imshow(cv2.cvtColor(bilateral,cv2.COLOR_BGR2RGB))
# axs[1][1].set_title('Bilateral Blur')
# cv2.imwrite('Bilateral_blur.png',bilateral)
# plt.show()
# #denoising
# denoise=cv2.fastNlMeansDenoisingColored(image,None,6,6,7,21)
# plt.imshow(cv2.cvtColor(denoise,cv2.COLOR_BGR2RGB))
# plt.title('Fast Means Denoising')
# plt.show()
# # There are 4 variations of Non-Local Means Denoising:
# # -cv2.fastNlMeansDenoising() - works with a single grayscale images
# # -cv2.fastNlMeansDenoisingColored() - works with a color image.
# # -cv2.fastNlMeansDenoisingMulti() - works with image sequence captured in short period of time (grayscale images)
# # -cv2.fastNlMeansDenoisingColoredMulti() - same as above, but for color images.

#.......................................................................................................................................................
# # SHARPENING
#.......................................................................................................................................................
# row,col=1,2
# fig,axs=plt.subplots(row,col,figsize=(15,10))
# fig.tight_layout()
# axs[0].imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
# axs[0].set_title('Original Image')
# sharpe_kernel=np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
# sharpe_img=cv2.filter2D(image,-1,sharpe_kernel)
# axs[1].imshow(cv2.cvtColor(sharpe_img,cv2.COLOR_BGR2RGB))
# axs[1].set_title('Sharpened Image')
# plt.show()
# cv2.imwrite("Sharpened_image.png",sharpe_img)

#.......................................................................................................................................................
# # THRESHOLDING, BINARIZATION, ADAPTIVE THRESHOLDING
#.......................................................................................................................................................
# row,col=2,3
# fig,axs=plt.subplots(row,col,figsize=(15,10))
# fig.tight_layout()
# axs[0][0].imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
# axs[0][0].set_title('Original Image')
# ret,thresh1=cv2.threshold(image,127,255,cv2.THRESH_BINARY)
# axs[0][1].imshow(cv2.cvtColor(thresh1,cv2.COLOR_BGR2RGB))
# axs[0][1].set_title('Threshold - binary')
# cv2.imwrite("thresh1.png",thresh1)
# ret,thresh2=cv2.threshold(image,127,255,cv2.THRESH_BINARY_INV)
# axs[0][2].imshow(cv2.cvtColor(thresh2,cv2.COLOR_BGR2RGB))
# axs[0][2].set_title('Threshold - binary inverse')
# cv2.imwrite("thresh2.png",thresh2)
# ret,thresh3=cv2.threshold(image,127,255,cv2.THRESH_TRUNC)
# axs[1][0].imshow(cv2.cvtColor(thresh3,cv2.COLOR_BGR2RGB))
# axs[1][0].set_title('Threshold - trunc')
# cv2.imwrite("thresh3.png",thresh3)
# ret,thresh4=cv2.threshold(image,127,255,cv2.THRESH_TOZERO)
# axs[1][1].imshow(cv2.cvtColor(thresh4,cv2.COLOR_BGR2RGB))
# axs[1][1].set_title('Threshold - to zero')
# cv2.imwrite("thresh4.png",thresh4)
# ret,thresh5=cv2.threshold(image,127,255,cv2.THRESH_TOZERO_INV)
# axs[1][2].imshow(cv2.cvtColor(thresh5,cv2.COLOR_BGR2RGB))
# axs[1][2].set_title('Threshold - to zero inverse')
# cv2.imwrite("thresh5.png",thresh5)
# plt.show()
# image1=cv2.imread(r'C:\Users\sraka\Downloads\e1.jpg',0)
# row,col=2,2
# fig,axs=plt.subplots(row,col,figsize=(15,10))
# fig.tight_layout()
# cv2.imwrite("Original_Image_for_adaptive_thresh.png",image1)
# ret,thresh1=cv2.threshold(image1,127,255,cv2.THRESH_BINARY)
# axs[0][0].imshow(cv2.cvtColor(thresh1,cv2.COLOR_BGR2RGB))
# axs[0][0].set_title('Threshold - binary')
# cv2.imwrite("thresh1_binary.png",thresh1)
# img1=cv2.GaussianBlur(image1,(3,3),0)
# thresh = cv2.adaptiveThreshold(img1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 5) 
# axs[0][1].imshow(cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB))
# axs[0][1].set_title('Adaptive Mean Thresholding')
# cv2.imwrite("adaptive_thresh2.png",thresh)
# _,th=cv2.threshold(img1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# axs[1][0].imshow(cv2.cvtColor(th,cv2.COLOR_BGR2RGB))
# axs[1][0].set_title("Otsu's Threshold")
# cv2.imwrite("adaptive_thresh3_otsu.png",th)
# img1=cv2.GaussianBlur(image1,(5,5),0)
# _,thresh4=cv2.threshold(img1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# axs[1][1].imshow(cv2.cvtColor(thresh4,cv2.COLOR_BGR2RGB))
# axs[1][1].set_title("Gaussian Threshold")
# cv2.imwrite("adaptive_thresh4_Gaussian.png",thresh4)
# plt.show()

#.......................................................................................................................................................
# #DILATION, EROSION, OPENING AND CLOSING
#.......................................................................................................................................................
# row,col=2,2
# image=cv2.imread(r'C:\Users\sraka\Downloads\e1.jpg')
# fig,axs=plt.subplots(row,col,figsize=(15,10))
# fig.tight_layout()
# kernel=np.ones((5,5),dtype='uint8')
# eroded=cv2.erode(image,kernel,iterations=1)
# axs[0][0].imshow(cv2.cvtColor(eroded,cv2.COLOR_BGR2RGB))
# axs[0][0].set_title('Eroded')
# cv2.imwrite("eroded.png",eroded)
# dilated=cv2.dilate(image,kernel,iterations=1)
# axs[0][1].imshow(cv2.cvtColor(dilated,cv2.COLOR_BGR2RGB))
# axs[0][1].set_title('Dilated')
# cv2.imwrite("dilated.png",dilated)
# opening=cv2.morphologyEx(image,cv2.MORPH_OPEN,kernel)
# axs[1][0].imshow(cv2.cvtColor(opening,cv2.COLOR_BGR2RGB))
# axs[1][0].set_title('opening')
# cv2.imwrite("opening.png",opening)
# closing=cv2.morphologyEx(image,cv2.MORPH_CLOSE,kernel)
# axs[1][1].imshow(cv2.cvtColor(closing,cv2.COLOR_BGR2RGB))
# axs[1][1].set_title('closing')
# cv2.imwrite("closing.png",closing)
# plt.show()

#.......................................................................................................................................................
# #EDGE DETECTION
#.......................................................................................................................................................
# sobel_x=cv2.Sobel(image,cv2.CV_64F,0,1,ksize=5)
# sobel_y=cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5)
# sobel=cv2.bitwise_or(sobel_x,sobel_y)
# laplacian=cv2.Laplacian(image,cv2.CV_64F)
# row,col=1,2
# fig,axs=plt.subplots(row,col,figsize=(15,10))
# fig.tight_layout()
# axs[0].imshow(cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
# axs[0].set_title('Original Image')
# cv2.imwrite('original_image.png', image)
# canny = cv2.Canny(image, 50, 120)
# axs[1].imshow(cv2.cvtColor(canny, cv2.COLOR_BGR2RGB))
# axs[1].set_title('canny')
# cv2.imwrite('canny.png', canny)
# plt.show()

#.......................................................................................................................................................
# # PERSPECTIVE AND AFFINE TRANSFORM
#.......................................................................................................................................................
# image1=cv2.imread(r'C:\Users\sraka\Downloads\bill.jpg')
# row, col = 1, 2
# fig, axs = plt.subplots(row, col, figsize=(15, 10))
# fig.tight_layout()
# axs[0].imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
# axs[0].set_title('Original RGB Colors')
# cv2.imwrite('original_rgb.jpg', image1)
# points_A=np.float32([[50,50], [270,70],[20,400], [275,410]])
# points_B = np.float32([[0,0], [420,0], [0,594], [420,594]])
# M=cv2.getPerspectiveTransform(points_A,points_B)
# warp=cv2.warpPerspective(image1,M,(420,594))
# axs[1].imshow(cv2.cvtColor(warp, cv2.COLOR_BGR2RGB))
# axs[1].set_title('warp perspective')
# cv2.imwrite('warp_perspective.png', warp)
# plt.show()
# row, col = 1, 2
# fig, axs = plt.subplots(row, col, figsize=(15, 10))
# fig.tight_layout()
# axs[0].imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
# axs[0].set_title('Original RGB Colors')
# points_A=np.float32([[50,50], [270,70],[275,410]])
# points_B = np.float32([[0,0], [420,0], [420,594]])
# M=cv2.getAffineTransform(points_A,points_B)
# warp1=cv2.warpAffine(image1,M,(420,594))
# axs[1].imshow(cv2.cvtColor(warp1, cv2.COLOR_BGR2RGB))
# axs[1].set_title('warp affine')
# cv2.imwrite('warp_affine.png', warp1)
# plt.show()

#.......................................................................................................................................................
#.......................................................................................................................................................
