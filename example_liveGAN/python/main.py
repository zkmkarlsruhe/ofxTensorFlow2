# -*- coding: utf-8 -*-

import os, argparse
from dataloader import Dataloader
from model import Model
import sys
from utils import check_dir

import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


def parse_args():
	desc = 'TensorFlow 2.0 implementation of Deep Convolutional Generative Adversarial Network (DCGAN)'
	parser = argparse.ArgumentParser(description=desc)

	parser.add_argument('--dataset_name', type=str, default='celeba')
	parser.add_argument('--phase', type=str, default='tfrecord', choices=('tfrecord', 'train', 'test'))
	parser.add_argument('--img_size', type=int, default=256)
	parser.add_argument('--img_nc', type=int, default=3)
	parser.add_argument('--z_dim', type=int, default=128)

	parser.add_argument('--batch_size', type=int, default=100)
	parser.add_argument('--iteration', type=int, default=100000)
	parser.add_argument('--log_freq', type=int, default=1000)
	parser.add_argument('--sample_freq', type=int, default=1000)
	parser.add_argument('--save_freq', type=int, default=1000)
	parser.add_argument('--output_dir', type=str, default='output')
	parser.add_argument('--log_dir', type=str, default='log')
	parser.add_argument('--sample_dir', type=str, default='sample')
	parser.add_argument('--save_dir', type=str, default='model')
	parser.add_argument('--result_dir', type=str, default='result')

	parser.add_argument('--lr', type=float, default=0.0002)
	parser.add_argument('--gan_type', type=str, default='vanilla', choices=('vanilla', 'lsgan', 'hinge'))

	args = parser.parse_args()
	check_dir(args.output_dir)
	args.output_dir = os.path.join(args.output_dir, f'DCGAN_{args.dataset_name}')
	check_dir(args.output_dir)
	args.log_dir = os.path.join(args.output_dir, args.log_dir)
	check_dir(args.log_dir)
	args.sample_dir = os.path.join(args.output_dir, args.sample_dir)
	check_dir(args.sample_dir)
	args.save_dir = os.path.join(args.output_dir, args.save_dir)
	check_dir(args.save_dir)
	args.result_dir = os.path.join(args.output_dir, args.result_dir)
	check_dir(args.result_dir)

	return args

if __name__ == '__main__':
	args = parse_args()
	if args.phase == 'tfrecord':
		print('Converting data to tfrecord...')
		Dataloader(args)
		print('Convert finished...')

	else:
		model = Model(args)
		model.build_model()

		if args.phase == 'train':
			print('Training...')
			model.train()
			print('Train finished...')
		
		elif args.phase == 'test':
			print('Testing...')
			model.test()
			print('Test finished...')
