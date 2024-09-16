import cv2
import numpy as np

class whiteBalance:
    def __init__(self, image):
        self.image = image

    def wbCorrection(self):
        img_float = self.image.astype(np.float32) / 255.0

        avg_b = np.mean(img_float[:,:,0])
        avg_g = np.mean(img_float[:,:,1])
        avg_r = np.mean(img_float[:,:,2])

        avg = (avg_b + avg_g + avg_r) / 3
        scale_b = avg / avg_b
        scale_g = avg / avg_g
        scale_r = avg / avg_r

        img_float[:,:,0] *= scale_b
        img_float[:,:,1] *= scale_g
        img_float[:,:,2] *= scale_r

        img_float = np.clip(img_float, 0, 1)
        img_float = (img_float * 255).astype(np.uint8)

        return img_float