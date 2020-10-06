import torch
import os
from torch.nn.utils.rnn import pad_sequence
import numpy as np


def construct_backward_sample(beam_out, beam_ind, sample, tgt_dict, prefix_source):
	target = torch.ones_like(sample['net_input']['src_tokens']) * tgt_dict.pad_index
	prev_output_seq = torch.ones_like(target) * tgt_dict.pad_index

	for i, leng in enumerate(sample['net_input']['src_lengths']):
		target[i, :leng] = sample['net_input']['src_tokens'][i, -leng:]
		prev_output_seq[i, 0] = tgt_dict.eos()
		prev_output_seq[i, 1:leng] = target[i, :leng - 1]

	if prefix_source:
		try:
			prefix_tokens = torch.stack([beam[beam_ind]['tokens'] for beam in beam_out], dim=0)
		except:
			print('gothere!')
		src_tokens = torch.cat(
			[prefix_tokens, torch.ones_like(prefix_tokens)[:, -2:-1] * tgt_dict.eos()],
			dim=1
		)
		# src_tokens = prefix_tokens
	else:
		src_tokens = torch.flip(
			pad_sequence([
				torch.flip(beam[beam_ind]['tokens'], [-1])[:-sample['prefix_length']]
				for beam in beam_out
			], padding_value=tgt_dict.pad_index, batch_first=True),
			[-1]
		)

	reverse_sample = {
		'nsentences': -1,
		'ntokens': -1,
		'net_input': {
			'src_tokens': src_tokens,
			'src_lengths': torch.LongTensor(
				[len(beam[beam_ind]['tokens']) - sample['prefix_length'] for beam in beam_out]
				if not prefix_source else
				[sample['prefix_length'] + 1] * (src_tokens.shape[0])
			),
			'prev_output_tokens': prev_output_seq
		},
		'target': target,

	}
	return reverse_sample


def update_beams_mmi(beam_out, scores, backward_scores, score_label):
	for i in range(len(beam_out)):

		for sc, bwd_sc, beam in zip(scores[i], backward_scores[i], beam_out[i]):
			beam[f'bidi_{score_label}_score'] = beam['score'] = sc
			beam[f'backward_{score_label}_score'] = bwd_sc
		try:
			beam_out[i] = [x for _, x in sorted(zip(scores[i], beam_out[i]), key=lambda x: -x[0])]
		except:
			import pdb;pdb.set_trace()

flatten = lambda l: [item for sublist in l for item in sublist]

def rerank_bidi(beam_out, model_dict, direction, args, sample, tgt_dict):
	forward_scores = np.array([[item['score'] for item in beam] for beam in beam_out])
	backward_scores = np.zeros_like(forward_scores)

	t2s = lambda tensor: tgt_dict.string(tensor, args.remove_bpe, escape_unk=True)

	args.sentence_avg = 1
	criterion = model_dict[direction].task.build_criterion(args)

	for beam_ind in range(forward_scores.shape[1]):
		reverse_sample = construct_backward_sample(beam_out, beam_ind, sample, tgt_dict,
												   prefix_source=('prefix' in direction))

		backward_beams = model_dict[direction].task.inference_step(
			model_dict[direction].generator, [model_dict[direction].model],
			reverse_sample, reverse_sample['target'])

		for b_i, bbeam in enumerate(backward_beams):
			backward_scores[b_i,beam_ind] = bbeam[0]['score']


	_normalize_rows = lambda mat:  ((mat.T - mat.mean(axis=1).T) / mat.std(axis=1).T).T

	label="prefix" if "prefix" in direction else "sequence"
	lbda = (args.prefix_bidi_lambda if label=='prefix' else args.sequence_bidi_lambda)

	scores = forward_scores + lbda * backward_scores
	update_beams_mmi(beam_out, scores, backward_scores, score_label=label)

	import random
	### TODO remove sequence
	if args.very_verbose and random.random() < 1 :

		with open(os.path.join(args.results_path, 'bidi_beams.log'),'a',buffering=1) as f:
			rand_ind = 1
			print(tgt_dict.string(
				sample['net_input']['src_tokens'][rand_ind],
				args.remove_bpe, escape_unk=True)
			, file=f)
			# print(direction)
			for beam_item in beam_out[rand_ind]:
				print(
					"{:<.6f},{:<.6f},{:<.6f}\t{}".format(
						beam_item[f'bidi_{label}_score'],
						beam_item[f'forward_{label}_score'],
						beam_item[f'backward_{label}_score'],
						tgt_dict.string(beam_item['tokens'], args.remove_bpe, escape_unk=True)
					), file=f
				)

	return beam_out