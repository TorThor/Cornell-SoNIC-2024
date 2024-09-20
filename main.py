import cv2
import matplotlib.pyplot as plt
from imgProcessing.imgData import imgData
from imgProcessing.imgPro import imgPro
from imgProcessing.whiteBalance import whiteBalance

# Setup image variables
image_name = input("Name of image: ")
image_path = imgData(image_name, "images\preProImg").photoFinder()
image = cv2.imread(image_path)

# Create processed imgs by calling class methods and save them to proImg
upscale_img = imgPro(image).upscaler()
denoise_img = imgPro(image).denoise()
clahe_img = imgPro(image).clahe()
gaussianBlur_img = imgPro(image).gaussian_blur()
wb_img = whiteBalance(image).wbCorrection()

# Save gaussianBlur_img & wb_img
imgData(image_name, "images\proImg").photoSaver("_wb", wb_img)
imgData(image_name, "images\proImg").photoSaver("_gaussianBlur", gaussianBlur_img)

# Display imgs for fun ;), though only gaussianBlur_img and wb_img used for threatsLLaVA and depthEstimator
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.title("Original")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.subplot(2, 3, 2)
plt.title("x2 Linear Upscale")
plt.imshow(cv2.cvtColor(upscale_img, cv2.COLOR_BGR2RGB))

plt.subplot(2, 3, 3)
plt.title("Denoised")
plt.imshow(cv2.cvtColor(denoise_img, cv2.COLOR_BGR2RGB))

plt.subplot(2, 3, 4)
plt.title("CLAHE")
plt.imshow(cv2.cvtColor(clahe_img, cv2.COLOR_BGR2RGB))

plt.subplot(2, 3, 5)
plt.title("Gaussian Blur")
plt.imshow(cv2.cvtColor(gaussianBlur_img, cv2.COLOR_BGR2RGB))

plt.subplot(2, 3, 6)
plt.title("wb (for depthEstimator, applied separately)")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.tight_layout()
plt.show()