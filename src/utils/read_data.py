from typing import Iterator
from tqdm import tqdm
from torch.utils.data.dataset import IterableDataset
from typing import Iterable
from itertools import islice
from src.utils import *


class SentenceDataset(IterableDataset):

	def __init__(self, file_path, every=1, displace=0, only_codes=None):
		self.file_path = file_path
		self.every = every
		self.displace = displace
		if only_codes is not None:
			self.codes = only_codes

	def parse_file(self, file_path):
		return read_sentences(
			file_path,
			codes=(self.codes if hasattr(self, 'codes') else None)
		)

	def get_stream(self, file_path):
		return self.parse_file(file_path)

	def __iter__(self) -> Iterable:
		stream = self.get_stream(self.file_path)
		return islice(stream, self.displace, None, self.every)


def blocks(files, size=65536):
	while True:
		b = files.read(size)
		if not b: break
		yield b


def get_num_lines(file: str):
	with open(file, "r", encoding="utf-8", errors='ignore') as f:
		return (sum(bl.count("\n") for bl in blocks(f)))


def read_sentences(path: str, progbar=True, codes=None, encoding='utf-8') -> Iterator[str]:
	if progbar: print(f'reading lines from {path}')
	numlines = get_num_lines(path)
	with open(path, 'r', encoding=encoding, errors='replace') as f:
		# if progbar:
		for l in (tqdm(f, total=numlines) if progbar else f):
			if codes:
				y=False
				for c in codes:
					if l.startswith(c):
						y=True
						break
			else:
				y=True
			if y:
				yield l.strip()
		# else:
		# 	for l in f:
		# 		if codes:
		# 			y = False
		# 			for c in codes:
		# 				if l.startswith(c):
		# 					y = True
		# 					break
		# 		else:
		# 			y = True
		# 		if y:
		# 			yield l.strip()
