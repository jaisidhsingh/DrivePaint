import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from collections import namedtuple



Label = namedtuple('Label', [
	'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
	'id'          , # An integer ID that is associated with this label.
	'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
	'category'    , # The name of the category that this label belongs to
	'categoryId'  , # The ID of this category. Used to create ground truth images
	'hasInstances', # Whether this label distinguishes between single instances or not
	'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
	'color'       , # The color of this label
])

predicted_id2label = {
    0: 'road', 1: 'sidewalk', 2: 'building', 3: 'wall', 4: 'fence',
	5: 'pole', 6: 'traffic light', 7: 'traffic sign', 8: 'vegetation',
	9: 'terrain', 10: 'sky', 11: 'person', 12: 'rider',	13: 'car',
	14: 'truck', 15: 'bus',	16: 'train', 17: 'motorcycle', 18: 'bicycle'
}

gt_database = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]


def make_gt_label2id():
	result = {}
	inference_labels = list(predicted_id2label.values())
	inference_ids = list(predicted_id2label.keys())
	N = len(inference_labels)

	for i in range(N):
		for item in gt_database:
			if item.name == inference_labels[i]:
				result[item.name] = {
					"color": item.color,
					"id": inference_ids[i],
				}
	
	return result


def convert_color_gt_image(image_path, output_folder, gt_label2id): # gt_label2id = make_gt_label2id()
	image_name = image_path.split("/")[-1]

	gt_image = Image.open(image_path).convert("RGB")
	gt_array = np.array(gt_image)

	result_shape = [gt_image.size[1], gt_image.size[0]]
	result_array = np.zeros(result_shape)

	for label in gt_label2id.keys():
		color = gt_label2id[label]["color"]
		color_array = np.array(color, dtype=np.uint8)
		mask4color = np.all(gt_array == color_array, axis=-1)

		id = gt_label2id[label]["id"]
		result_array[mask4color] = id
	
	result_array = np.uint8(result_array)
	converted_gt_image = Image.fromarray(result_array)
	converted_gt_image.save(os.path.join(output_folder, image_name))

