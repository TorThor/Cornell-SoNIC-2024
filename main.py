import os
import glob
import cv2
from imgPro import imgPro
import matplotlib as plt

def photoFinder(img_name, path):
    root_dir = os.path.dirname(os.path.abspath(__file__))
    allowed_extensions = ['.JPG', '.jpeg', '.jpg', '.png', '.PNG']
    image_path = None

    for ext in allowed_extensions:
        files = glob.glob(os.path.join(root_dir, path, img_name + ext))
        if files:
            image_path = files[0]
            break
    return image_path

image_path = photoFinder("nyc_traffic", "preProImg")
image = cv2.imread(image_path)
"""
cv2.imshow("window_name", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

"""
upscale_img = imgPro.upscaler(image)
denoise_img = imgPro.denoise(image)
clahe_img = imgPro.clahe(image)
guassianBlur_img = imgPro.gaussian_blur(image)

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
plt.imshow(cv2.cvtColor(guassianBlur_img, cv2.COLOR_BGR2RGB))

plt.tight_layout()
plt.show()
"""