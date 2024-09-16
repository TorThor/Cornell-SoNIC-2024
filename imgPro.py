import cv2
import numpy as np

class imgPro:
    def __init__(self, image):
        self.image = image
    
    def upscaler(self):
        upscaled_image = cv2.resize(self.image, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
        return upscaled_image

    def denoise(self):
        noise_std = np.std(self.upscaler())
        denoised_img = cv2.fastNlMeansDenoisingColored(self.image, None, noise_std / 4, 7, 21)
        return denoised_img
    
    def clahe(self):
        lab = cv2.cvtColor(self.denoise(), cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        lab_clahe = cv2.merge((l_clahe, a, b))
        bgr_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        return bgr_clahe
    
    def gaussian_blur(self):
        blurred_img = cv2.GaussianBlur(self.clahe(), (5, 5), 0)
        return blurred_img