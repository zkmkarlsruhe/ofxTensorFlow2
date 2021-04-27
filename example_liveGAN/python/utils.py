# -*- coding: utf-8 -*-

import numpy as np
import cv2, imageio, os

def check_dir(out_dir):
	if not os.path.exists(out_dir):
		os.mkdir(out_dir)

def imread(img_path, norm=True):
	img = imageio.imread(img_path)
	return img / 255. if norm else img

def imsave(save_path, img):
	imageio.imsave(save_path, img)

def mimsave(save_path, imgs, fps=10):
	imageio.mimsave(save_path, imgs, fps=fps)

def imresize(img, h, w, method='LINEAR'):
	if method == 'LINEAR':
		return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
	elif method == 'NEAREST':
		return cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)

def center_crop(img):
	h, w = img.shape[:2]
	if h >= w:
		return img[h // 2 - w // 2: h // 2 - w // 2 + w, :]
	else:
		return img[:, w // 2 - h // 2: w // 2 - h // 2 + h]

def imnorm(img):
	return (img - 0.5) * 2.

def imdenorm(img):
	return (img + 1.) / 2.

def montage(imgs):
	N, H, W, C = imgs.shape
	n = int(np.ceil(np.sqrt(N)))

	result = np.ones((n * H, n * W, C))
	for i in range(N):
		r, c = i // n, i % n
		result[r * H: (r + 1) * H, c * W: (c + 1) * W] = imgs[i]

	return result

def lerp_np(start, end, ratio):
	return start + (end - start) * np.clip(ratio, 0.0, 1.0)

def ceil(x):
	return int(np.ceil(x))

def floor(x):
	return int(np.floor(x))

def get_nonzero_region(img):
	non = np.nonzero(img)
	if len(non[0]) == 0 or len(non[1]) == 0:
		y0, y1, x0, x1 = 0, img.shape[0] - 1, 0, img.shape[1] - 1
	else:
		y0 = np.min(non[0])
		y1 = np.max(non[0])
		x0 = np.min(non[1])
		x1 = np.max(non[1])
	return y0, y1, (y0 + y1) // 2, x0, x1, (x0 + x1) // 2

def array_to_list(array):
	return np.reshape(array, [array.size]).tolist()