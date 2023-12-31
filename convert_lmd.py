"""convert the LMD dataset to masks.
"""
import os
from PIL import Image
import numpy as np
from glob import glob
import tqdm

def load_and_save_mask_for_one_image(img_name, mask_paths, category2code):
    """load the original masks, and combine them into one for each image.

    Args:
        img_name (str): the name of the RGB image.
    """
    origin_img = Image.open(os.path.join("data/localmatdb/images", img_name)).convert('RGB')
    segment_paths = [path for path in mask_paths if img_name in path]

    # get the category from path str like ./COCO_train2014_000000010073.jpg_food_mask.png
    segment_categories = []
    for seg_path in segment_paths:
        category = seg_path.split("_")[-2]
        if category not in segment_categories:
            segment_categories.append(category)
        else:
            not_refined_mask_path = [path for path in segment_paths if
                                     category in path and "refinedmask" not in path]
            for path in not_refined_mask_path:
                segment_paths.remove(path)

    # load the masks and merge them into one
    segment_img = np.full((origin_img.size[::-1]), 255, dtype=np.uint8)  # default all 255, means unknown
    for segment_path in segment_paths:
        temp_mask = np.asarray(Image.open(segment_path))
        # print(segment_path)
        assert temp_mask.shape == segment_img.shape
        # If multiple, test segment_img==255 to avoid overwriting.
        segment_img[(temp_mask != 0) & (segment_img == 255)] = category2code[segment_path.split("_")[-2]]

    segment_img = Image.fromarray(segment_img)

    # resize the image with min dimension as 512.
    W, H = origin_img.size
    if H <= W:
        new_size = (round(512 * W / H), 512)
    else:
        new_size = (512, round(512 * H / W))

    origin_img = origin_img.resize(new_size, Image.BILINEAR)
    segment_img = segment_img.resize(new_size, Image.NEAREST)

    name = img_name.split("/")[-1][:-4]
    
    origin_img.save("data/localmatdb/images_resized/{}.png".format(name))
    segment_img.save("data/localmatdb/masks_png/{}.png".format(name))

if __name__ == "__main__":
    """enumerate  the file names
    """
    mask_path = "data/localmatdb/masks_png"
    resized_image_path = "data/localmatdb/images_resized"
    if not os.path.isdir(mask_path):
        os.mkdir(mask_path)
    if not os.path.isdir(resized_image_path):
        os.mkdir(resized_image_path)
    image_names = os.listdir("data/localmatdb/images")
    mask_paths = glob("data/localmatdb/masks/**/*.png", recursive=True)
    category2code = {"asphalt": 0, "ceramic": 1, "concrete": 2, "fabric": 3, "foliage": 4,
                              "food": 5, "glass": 6, "metal": 7, "paper": 8, "plaster": 9, "plastic": 10,
                              "rubber": 11, "soil": 12, "stone": 13, "water": 14, "wood": 15}
    for img_name in tqdm.tqdm(image_names):
        load_and_save_mask_for_one_image(img_name, mask_paths, category2code)