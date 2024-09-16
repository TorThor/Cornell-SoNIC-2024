import os
import glob
import cv2


def photoFinder(img_name, path):
    img_dir = f"/content/drive/MyDrive/SoNIC/{path}"

    allowed_extensions = ['.JPG', '.jpeg', '.jpg', '.png', '.PNG']
    image_path = None
    original_extension = None

    for ext in allowed_extensions:
        files = glob.glob(os.path.join(img_dir, img_name + ext))
        if files:
            image_path = files[0]
            original_extension = ext
            break
    return image_path

image_path = photoFinder("VSCode", "preProImg")
image = cv2.imread(image_path)