import argparse
import torch
from tqdm import tqdm
import os
from PIL import Image, ImageDraw
import numpy as np
import cv2


def get_class_index(args):
	dataset = args.command.split("_")[0]
	class_name = args.command.split("_")[1]

	mapping = {
		"cityscapes": {
			'road': 0, 'sidewalk': 1, 'building': 2, 'wall': 3, 
			'fence': 4, 'pole': 5, 'traffic light': 6, 'traffic sign': 7, 
			'vegetation': 8, 'terrain': 9, 'sky': 10, 'person': 11, 
			'rider': 12, 'car': 13, 'truck': 14, 'bus': 15, 
			'train': 16, 'motorcycle': 17, 'bicycle': 18
		},
		"bdd": {
			'road': 0, 'sidewalk': 1, 'building': 2, 'wall': 3, 
			'fence': 4, 'pole': 5, 'traffic light': 6, 'traffic sign': 7, 
			'vegetation': 8, 'terrain': 9, 'sky': 10, 'person': 11, 
			'rider': 12, 'car': 13, 'truck': 14, 'bus': 15, 
			'train': 16, 'motorcycle': 17, 'bicycle': 18
		}
	}
	
	return mapping[dataset][class_name]

def make_binary(image_path, class_index):
	image = Image.open(image_path)
	image_array = np.array(image)
	binary_array = np.zeros(image_array.shape)

	class_index = np.uint8(class_index)
	present_mask = image_array == class_index
	absent_mask = image_array != class_index

	binary_array[present_mask] = np.uint8(1)
	binary_array[absent_mask] = np.uint8(0)

	return np.uint8(binary_array)

def test_overlay(image_path, x, y, w, h):
	image = Image.open(image_path)
	draw = ImageDraw.Draw(image)

	rect_coords = (x, y, x + w, y + h)
	draw.rectangle(rect_coords, outline="red", width=2)
	image.show()

def find_patches(image_path, class_index, area_threshold=50, hw_threshold=5):
	coordinates = []

	binary_array = make_binary(image_path, class_index)
	contours, _ = cv2.findContours(binary_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	for cont in contours:
		area = cv2.contourArea(cont)

		if area > area_threshold: # can't operate on too small patches (unclear)
			x, y, w, h = cv2.boundingRect(cont)

			if w > hw_threshold and h > hw_threshold: # can't operate on too small patches
				coordinates.append([x, y, w, h])

	return coordinates

def main(args):
	class_index = get_class_index(args)
	image_names = os.listdir(args.input_folder)
	image_paths = [os.path.join(args.input_folder, p) for p in image_names]

	result = {}	
	bar = tqdm(total=len(image_paths))

	for i in range(len(image_paths)):
		coordinates = find_patches(image_paths[i], class_index)
		result[image_names[i]] = coordinates
	
	save_name = args.command + "_patches.pt"
	save_path = os.path.join(args.output_folder, save_name)
	torch.save(result, save_path)

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--command", type=str, default="cityscapes_person")
	parser.add_argument("--input-folder", type=str, default="semseg_predictions")
	parser.add_argument("--output-folder", type=str, default="patchify_results")
	args = parser.parse_args()
	
	return args


if __name__ == "__main__":
	args = get_args()
	main(args)
