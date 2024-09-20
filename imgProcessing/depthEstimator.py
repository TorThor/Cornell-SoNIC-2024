import os
# from transformers import pipeline

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
"""
checkpoint = "vinvino02/glpn-nyu"
depth_estimator = pipeline("depth-estimation", model=checkpoint)

image_path = os.path.join(root_dir, "proImg", )

image_path = f"/content/drive/MyDrive/SoNIC_Team_4_Llava_Photos/proImgs/depth0_proImgs_wb.jpg"        # {img_name}_{location}_{custom}{original_extension}"
image = skimage.io.imread(image_path)
image = Image.fromarray(np.uint8(image)).convert("RGB")
pipelined_image=image
predictions = depth_estimator(image)
depth_image=predictions["depth"]
depth_image       # (location, img_name, custom, original_extension)
photoSave("wholeImgs", "depth", "depth", {original_extension})\
"""