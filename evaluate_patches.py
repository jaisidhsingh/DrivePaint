import numpy as np
import os
from PIL import Image
from .make_patches import make_binary

"""
FORMAT:
=======
raw scene filename: xyz.png
> signifies the scene that the DM input patch is a part of

pred class map filenane: xyz_pred.png

gt class map filename: xyz_gt.png

DM input filename: xyz_0.png
> 0 signifies the index of the patch in the total patches inside xyz.png

patch_meta: abc.pt
> a dict with keys as filenames and values as bboxes of the patches
"""

def get_patch_bbox(patch_filename, patch_metas):
	extension = patch_filename.split(".")[1]
	patch_index = int(patch_filename.split("__")[1].split(".")[0])

	scene_filename = patch_filename.split("__")[0] + f".{extension}"
	patch_bbox = patch_metas[scene_filename][patch_index]
	return patch_bbox

def locate_patch_pred(pred_filename, patch_bbox, class_index):
	x, y, w, h = patch_bbox

	pred_image = Image.open(pred_filename)
	pred_array = np.array(pred_image)

	patch_in_pred = pred_array[y:y+h, x:x+w]
	return make_binary(patch_in_pred, class_index)

def locate_patch_gt(gt_filename, patch_bbox, class_index):
	x, y, w, h = patch_bbox

	gt_image = Image.open(gt_filename)
	gt_array = np.array(gt_image)

	patch_in_gt = gt_array[y:y+h, x:x+w]
	return make_binary(patch_in_gt, class_index)

def iou_for_binary_maps(A, B):
	intersection = np.logical_and(A, B)
	union = np.logical_or(A, B)

	intersection_count = np.sum(intersection)
	union_count = np.sum(union)

	iou = intersection_count / union_count
	return iou

def evaluate_one_patch(patch_filename, patch_metas, configs):
	bbox = get_patch_bbox(patch_filename, patch_metas)

	extension = patch_filename.split(".")[1]
	scene_filename = patch_filename.split("__")[0] + f".{extension}"

	pred_filepath = os.path.join(configs.pred_folder, scene_filename)
	gt_filepath = os.path.join(configs.gt_folder, scene_filename)

	pred_bm = locate_patch_pred(pred_filepath, bbox, configs.class_index)
	gt_bm = locate_patch_gt(gt_filepath, bbox, configs.class_index)

	iou = iou_for_binary_maps(pred_bm, gt_bm)

	if iou > configs.threshold:
		return False
	else:
		return True





