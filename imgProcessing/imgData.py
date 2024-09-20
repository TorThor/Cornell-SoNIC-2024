import glob
import os
import cv2
import imgProcessing.depthEstimator as depthEstimator

class imgData:
    def __init__(self, img_name=None, path=None):
        self.img_name = img_name
        self.path = path

    # Useful function for finding path to an img given img_name and path from local root directory
    def photoFinder(self):
        allowed_extensions = ['.JPG', '.jpeg', '.jpg', '.png', '.PNG']
        image_path = None

        for ext in allowed_extensions:
            files = glob.glob(os.path.join(depthEstimator.root_dir, self.path, self.img_name + ext))
            if files:
                image_path = files[0]
                break
        return image_path

    # Useful function for saving processed imgs
    def photoSaver(self, extra, image):
        output_dir = os.path.join(depthEstimator.root_dir, self.path, self.img_name + extra) + ".jpg"
        cv2.imwrite(output_dir, image)