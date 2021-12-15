# -*- coding: utf-8 -*-

"""
@Author     : Jiangjie Chen
@Time       : 2020/6/8 22:17
@Contact    : jjchen19@fudan.edu.cn
@Description:
"""

import re
import os
import ujson as json
import tensorflow as tf


def args_to_shell(args):
	args_dict = vars(args)
	shell_args = ''
	for k, v in args_dict.items():
		if isinstance(v, bool):
			if v: shell_args += f'--{k} '
		else:
			if isinstance(v, list):
				v = ' '.join([str(x) for x in v])
			shell_args += f'--{k} {v} '
	return shell_args


def _is_proc_file(fname):
	return re.search('._\d+_proc$', fname) is not None


def _restore_fname_from_proc(fname):
	if _is_proc_file(fname):
		return '.'.join(fname.split('.')[:-1])
	else:
		return fname


def rename_fname_by_proc(fname: str, proc_num: int):
	if not _is_proc_file(fname):
		fname = fname + f'._{proc_num}_proc'
	return fname


def chunks(lst, n):
	"""Yield successive n-sized chunks from lst."""
	for i in range(0, len(lst), n):
		yield lst[i : i + n]


# def slice_dataset_json(in_fname, slice_num):
# 	with tf.io.gfile.GFile(in_fname) as fin:
# 		data = json.load(fin)
# 		sliced_data = chunks(data, slice_num)
# 		datasets = []
# 		for i in range(len(list(sliced_data))):
# 			proc_fname = rename_fname_by_proc(in_fname, i)
# 			with tf.io.gfile.GFile(proc_fname, 'w') as f:
# 				js = []
# 				for line in sliced_data[i]:
# 					js.append(line)
# 				json.dump(js, f)
# 			datasets.append(proc_fname)
# 	return datasets


def slice_filenames(in_fname, slice_num):
	sliced_f = []
	for i in range(slice_num):
		sliced_f.append(rename_fname_by_proc(in_fname, i))
	return sliced_f


def slice_dataset(in_fname, slice_num):
	'''
	:param in_fname:
	:param slice_num:
	:return: sliced dataset filenames
	'''
	with tf.io.gfile.GFile(in_fname) as fin:
		data = fin.readlines()
		_sliced_data = list(chunks(data, len(data) // slice_num))
		if len(_sliced_data) == slice_num + 1:  # loose ends
			sliced_data = _sliced_data[:slice_num]
			sliced_data[-1] += _sliced_data[-1]
		else:
			sliced_data = _sliced_data
		datasets = []
		for i in range(len(list(sliced_data))):
			proc_fname = rename_fname_by_proc(in_fname, i)
			with tf.io.gfile.GFile(proc_fname, 'w') as f:
				for line in sliced_data[i]:
					f.write(line)
			datasets.append(proc_fname)
	return datasets


def union_multiproc_files(files, overwrite=False):
	real_out_fname = None
	for i, file in enumerate(files):
		if not _is_proc_file(file):
			raise FileNotFoundError(file)
		else:
			_out_fname = _restore_fname_from_proc(file)
			if i > 0 and _out_fname != real_out_fname:
				raise ValueError(file, real_out_fname)
			real_out_fname = _out_fname

	if real_out_fname is None:
		raise FileNotFoundError(real_out_fname)

	if tf.io.gfile.exists(real_out_fname) and not overwrite:
		print(f'Skip {real_out_fname}, as it already exists.')
	else:
		with tf.io.gfile.GFile(real_out_fname, 'w') as fo:
			for file in files:
				if _is_proc_file(file):
					with tf.io.gfile.GFile(file) as f:
						data = f.readlines()
						for line in data:
							fo.write(line)
	print(f'{files} united into {real_out_fname}.')
	return real_out_fname


def clean_multiproc_files(files):
	for file in files:
		if _is_proc_file(file):
			if tf.io.gfile.exists(file):
				print(f'Removing {file}...')
				tf.io.gfile.remove(file)
			else:
				print(f'Removing {file}, but does not exists.')


if __name__ == '__main__':
	test_file = 'cjjpy.py'
	sliced_files = slice_dataset(test_file, 2)
	file = union_multiproc_files(sliced_files)
	clean_multiproc_files(sliced_files)