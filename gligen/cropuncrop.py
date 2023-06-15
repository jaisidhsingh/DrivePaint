import os
import pickle
import cv2
from PIL import Image


def crop_around_mask(base_image, mask_coordinates, crop_size=512):
    """
    make a 512x512 crop around the bounding box. given bbox coordinates
    for the uncropped image, make a crop around the bounding box location
    and present new coordinates for the bounding box in the cropped image.
    """
    image_width = base_image.shape[1]
    image_height = base_image.shape[0]

    (x_0, y_0, x, y) = mask_coordinates
    mask_width = x - x_0
    mask_height = y - y_0   
    x_mid, y_mid = [(x+x_0)/2, (y+y_0)/2]

    # if crop and mask centers coincide
    if image_width - x_mid >= crop_size//2 and image_height - y_mid >= crop_size//2:
        crop_coordinates_start = [int(x_mid - crop_size//2), int(y_mid - crop_size//2)]
        crop_coordinates_end = [int(x_mid + crop_size//2), int(y_mid + crop_size//2)]

        crop_coordinates = crop_coordinates_start + crop_coordinates_end

        cropped_image = base_image[crop_coordinates_start[1]: crop_coordinates_end[1], 
            crop_coordinates_start[0]: crop_coordinates_end[0]
        ]

        cropped_mask_bbox = [
            (crop_size - mask_width)//2,
            (crop_size - mask_height)//2,
            (crop_size + mask_width)//2,
            (crop_size + mask_height)//2
        ]

    # if crop and mask centers do not coincide 
    elif image_width - x_mid < crop_size//2 or image_height - y_mid < crop_size//2:
        crop_coordinates_start = [int(image_width - crop_size), int(image_height - crop_size)]
        crop_coordinates_end = [int(image_width + crop_size), int(image_height + crop_size)]

        crop_coordinates = crop_coordinates_start + crop_coordinates_end

        cropped_image = base_image[crop_coordinates_start[1]: crop_coordinates_end[1], 
            crop_coordinates_start[0]: crop_coordinates_end[0]
        ]

        center_offset_x = x_mid - image_width + crop_size//2
        center_offset_y = y_mid - image_height + crop_size//2

        cropped_mask_bbox = [
            crop_size//2 + center_offset_x - mask_width//2,
            crop_size//2 + center_offset_y - mask_height//2,
            crop_size//2 + center_offset_x + mask_width//2,
            crop_size//2 + center_offset_y + mask_height//2,
        ]

    cropped_mask_bbox = [x/crop_size for x in cropped_mask_bbox]
    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cropped_image).convert('RGB'), cropped_mask_bbox, crop_coordinates

def uncrop(base_image, cropped_image, crop_coordinates):
    """
    the input image are numpy arrays
    """
    tmp = base_image
    (x_0, y_0, x, y) = crop_coordinates
    tmp[y_0:y, x_0:x] = cropped_image

    tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB) 
    return Image.fromarray(tmp).convert('RGB')
 

def test():
    base_image_path = os.path.join("./data", "cityscapes_0.jpg")
    base_image = cv2.imread(base_image_path) 
    mask_bbox = [1173, 428, 1228, 505]

    cropped_returns = crop_around_mask(base_image, mask_bbox)
    uncropped_image = uncrop(base_image, cropped_returns[0], cropped_returns[2])
    