from DrivePaint.gligen.cropuncrop import crop_around_mask, uncrop
from DrivePaint.gligen.image_inpainting import run_inpainting
from types import SimpleNamespace

import os
import cv2
from PIL import Image
import numpy as np


def test(
    base_image_path="./data/cityscapes_0.jpg",
    mask_bbox=[1173, 428, 1228, 505],
    inpainting_content_path="./data/stroller_0.jpg"
):
    base_image = cv2.imread(base_image_path) 
    cropped_returns = crop_around_mask(base_image, mask_bbox)

    cropped_image = cropped_returns[0]
    mask_in_cropped_image = cropped_returns[1]
    crop_coordinates = cropped_returns[2]

    cropped_image.save("./data/cropped_0.png")

    args = SimpleNamespace(**{})
    args.base_image_path = "./data/cropped_0.png"
    args.inpainting_masks = [mask_in_cropped_image]
    args.inpainting_content = [inpainting_content_path]

    inpainting_result = run_inpainting(args)[0]
    inpainted_cropped_image = cv2.cvtColor(
        np.array(inpainting_result),
        cv2.COLOR_BGR2RGB
    )

    final = uncrop(base_image, inpainted_cropped_image, crop_coordinates)
    final.save("./results_testing/final.png")

test()
