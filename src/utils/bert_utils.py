global Device
from typing import List, Tuple, Iterator
import torch
from collections import namedtuple


class InputFeatures(object):
	"""A single set of features of data."""

	def __init__(self, tokens, input_ids, input_mask, input_type_ids):
		# self.unique_id = unique_id
		self.tokens = tokens
		self.input_ids = input_ids
		self.input_mask = input_mask
		self.input_type_ids = input_type_ids


def construct_bert_input(sent_batch: List[str], tokenizer, seq_len: int):
	features = []
	for text in sent_batch:
		tokens = ['[CLS]'] + tokenizer.tokenize(text)[:seq_len - 2] + ['[SEP]']
		input_ids = tokenizer.convert_tokens_to_ids(tokens)
		input_type_ids = [0] * len(input_ids)
		input_mask = [1] * len(input_ids)
		while len(input_ids) < seq_len:
			input_ids.append(0)
			input_mask.append(0)
			input_type_ids.append(0)

		assert len(input_ids) == seq_len
		assert len(input_mask) == seq_len
		assert len(input_type_ids) == seq_len

		features.append(
			InputFeatures(
				tokens=tokens,
				input_ids=input_ids,
				input_mask=input_mask,
				input_type_ids=input_type_ids))

	return features


BERTInput = namedtuple('BERTInput', ['ids', 'mask'])


def collate_features(sent_batch: Iterator[str], tokenizer, seq_len: int) -> Tuple[List[str], BERTInput]:
	features = construct_bert_input(sent_batch, tokenizer, seq_len)

	all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
	all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
	all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

	return sent_batch, BERTInput(all_input_ids, all_input_mask)


def extract_features(bert_input: BERTInput, model, device):
	input_ids = device.move(bert_input.ids)
	input_mask = device.move(bert_input.mask)

	_, pooled_output = model(input_ids, token_type_ids=None, attention_mask=input_mask)

	return device.to_numpy(pooled_output)
