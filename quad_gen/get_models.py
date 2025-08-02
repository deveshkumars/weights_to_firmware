#!/usr/bin/env python3

import argparse
import csv
import joblib
import os
import re
import sys
import shutil
from pathlib import Path
from typing import List, Optional

# Import TensorFlow
import tensorflow as tf
import yaml

import quad_gen.gaussian_mlp as mlp


def subdir(root_dir: str) -> List[str]:
	"""
	Return all subdirectories as a list
	Args:
		root_dir [str]: a root directory
	Returns:
		List[str]: list of subdirectory paths
	"""
	return [sub_f.path for sub_f in os.scandir(root_dir) if sub_f.is_dir()]


def read_txt_to_get_dirs(root_dir: str, txt: str) -> List[str]:
	"""
	Read a txt file containing directories to models
	Assuming that all the directories are sub directories
	(i.e. the full path to a model is root + '/' + sub_dir)
	"""
	sub_dirs = []
	root_dir = root_dir.strip().rstrip(os.sep)
	with open(txt, 'r') as f:
		for line in f:
			line = line.strip().lstrip(os.sep)
			full_path = f"{root_dir}/{line}"
			print(full_path)
			assert os.path.isdir(full_path), f"Directory does not exist: {full_path}"
			sub_dirs.append(full_path)
	return sub_dirs


def analyze_seeds(experiment: str) -> str:
	"""
	Find the seed directory with the highest average reward
	Args:
		experiment [str]: root directory of a single experiment containing multiple seeds
	Returns:
		str: the directory of the seed with the highest average reward
	"""
	assert os.path.isdir(experiment), f"Experiment directory does not exist: {experiment}"

	seeds = subdir(experiment)

	highest_reward = -float('inf')
	target_seed = ""
	best_seed = ""
	
	for seed_dir in seeds:
		# check if it is a seed directory
		seed_dir_split = seed_dir.split('/')
		if not re.search(r'^seed_*', seed_dir_split[-1]):
			print(f'Experiment {seed_dir} has seed folder that is named incorrectly... terminating...')
			sys.exit(1)
		else:
			progress_file = os.path.join(seed_dir, 'progress.csv')
			if not os.path.exists(progress_file):
				print(f'Progress file not found in {seed_dir}, skipping...')
				continue
				
			with open(progress_file, 'r') as csvfile:
				progress_reader = csv.DictReader(csvfile)
				rows = list(progress_reader)
				if not rows:
					print(f'No data found in progress.csv for {seed_dir}, skipping...')
					continue
					
				main_reward_latest = rows[-1]['rewards/rew_main_avg']
				if highest_reward <= float(main_reward_latest):
					target_seed = seed_dir
					best_seed = seed_dir_split[-1]
					highest_reward = float(main_reward_latest)

	if not target_seed:
		raise ValueError(f"No valid seed directories found in {experiment}")

	print(f'Best seed: {best_seed}')
	return target_seed


def save_result(model_dir: str, out_dir: str, osi: bool = False, absolute_path: bool = False) -> None:
	"""
	Save the result of a model to a directory
	Args:
		model_dir [str]: the directory containing the model
		out_dir [str]: the root directory of which the model should be saved
		osi [bool]: indicates whether the model is an osi
		absolute_path [bool]: (default False) indicated whether the out_dir 
			has been modified to the desired sub location
	"""
	model_dir = model_dir.rstrip(os.sep)
	out_dir = out_dir.rstrip(os.sep)
	
	if not absolute_path:
		# the out_dir is still the out_dir provided at the command line
		# try to append the correct sub_dir to it
		desired_sub_p = model_dir.split('/')[-5:]
		desired_sub_p = '/'.join(desired_sub_p)
		out_dir = f"{out_dir}/{desired_sub_p}"

	# Create output directory
	os.makedirs(out_dir, exist_ok=True)

	# Copy params.pkl file
	params_src = os.path.join(model_dir, 'params.pkl')
	params_dst = os.path.join(out_dir, 'params.pkl')
	shutil.copyfile(params_src, params_dst)
	
	# shutil.copyfile(model_dir + '/config.yml', out_dir + '/config.yml')

	# TensorFlow 2.x compatible session handling
	print(f"Extracting parameters from file {params_src} ...")
	pkl_params = joblib.load(params_src)
	
	# Handle the tuple structure: (running_stats, policy_params, value_params)
	if isinstance(pkl_params, tuple) and len(pkl_params) >= 2:
		# The policy is the second element in the tuple
		policy = pkl_params[1]
	else:
		# Fallback to the original dictionary structure
		policy = pkl_params['policy']

	# For TensorFlow 2.x, we need to handle the session differently
	# The mlp.generate function expects a session, so we'll create a TF 1.x compatible session
	# This maintains backward compatibility with the existing mlp module
	tf.compat.v1.disable_eager_execution()
	with tf.compat.v1.Session() as sess:
		mlp.generate(policy, sess, f"{out_dir}/network_evaluate.c")
	print(f"C code generated successfully: {out_dir}/network_evaluate.c")


def copy_by_best_seed(root_dir: str, out_dir: str) -> None:
	"""
	Copy models by selecting the best seed from each experiment
	Args:
		root_dir [str]: root directory containing experiments
		out_dir [str]: output directory for saved models
	"""
	print(f'Searching root {root_dir} ...')
	print('================================')
	subdirs = subdir(root_dir)

	for experiment in subdirs:
		print(f'Searching subdir {experiment} ... Analyzing seeds')
		# grab the seed with the highest average reward
		target_seed = analyze_seeds(experiment)
		save_result(target_seed, out_dir)


def copy_by_txt(root_dir: str, out_dir: str, txt: str) -> None:
	"""
	Copy the models specified in a txt file
	All the models must be located under the root_dir
	Args:
		root_dir [str]: the root directory
		out_dir [str]: the output directory [will create one if it doesn't exist]
		txt [str]: the txt file specifying the model relative directories
	"""
	print(f'Searching root {root_dir} ...')
	print('================================')

	subdirs = read_txt_to_get_dirs(root_dir, txt)
	for experiment in subdirs:
		print(f'Copying params.pkl from {experiment} to {out_dir}...')
		save_result(experiment, out_dir)


def traverse_root(root_dir: str, out_dir: str) -> None:
	"""
	Recursively search for pickle files of models and 
	convert the model if found

	Args:
		root_dir [str]: the root directory
		out_dir [str]: the output directory [will create one if it doesn't exist]
	"""
	subdirs = subdir(root_dir)
	for path in subdirs:
		path = path.rstrip(os.sep)
		params_file = os.path.join(path, 'params.pkl')
		if os.path.isfile(params_file):
			# -5 is picked appropriately
			save_path = '/'.join([i for i in path.split('/')[-5:]])
			save_path = f"{out_dir.rstrip(os.sep)}/{save_path}"
			print(f'Copying params.pkl from {path} to {save_path}...')
			save_result(path, save_path, absolute_path=True)
		else:
			traverse_root(path, out_dir)


def main(args: argparse.Namespace) -> None:
	"""Main function to handle different modes of operation"""
	if args.mode == 0:
		if not args.txt:
			raise ValueError("Mode 0 requires a txt file to be specified with -txt")
		copy_by_txt(args.root_dir, args.out_dir, args.txt)
	elif args.mode == 1:
		copy_by_best_seed(args.root_dir, args.out_dir)
	elif args.mode == 2:
		traverse_root(args.root_dir, args.out_dir)
	else:
		raise ValueError(f"Invalid mode: {args.mode}. Must be 0, 1, or 2.")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
	parser.add_argument(
		'mode',
		type=int,
		default=2,
		help='select a mode to copy file.\n'
			 '0: a txt file with dirs\n'
			 '1: a root where all the experiments are stored and select the best seeds.\n'
			 '2: a root dir where all the subdirs that contain plk file will be copied.\n', 
	)

	parser.add_argument(
		'root_dir',
		type=str,
		help='Root dir of the experiments'
	)

	parser.add_argument(
		'out_dir', 
		type=str,
		help='dir to save the experiments'
	)

	parser.add_argument(
		'-txt',
		type=str,
		help='txt file that contains all the models'
	)

	args = parser.parse_args() 

	main(args)