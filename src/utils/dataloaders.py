from string import punctuation
from torch.utils.data import DataLoader


def first_lower(s):
	if len(s) == 0:
		return s
	elif s[0] == 'I' and s[1] == ' ':
		return s
	else:
		return s[0].lower() + s[1:]

def first_upper(s):
	if len(s) == 0:
		return s
	else:
		return s[0].upper() + s[1:]

def well_formed_sentence(s):
	ret = s.strip()
	ret = ret.replace('  ', ' ')
	ret = first_upper(ret)
	if ret[-1] not in punctuation:
		ret = ret + '.'
	ret = ret.replace(' i ', ' I ')
	return ret

def model_output_sentence(s):
	ret = s.strip()
	ret = ret.replace('  ', ' ')
	ret = first_lower(ret)
	if ret[-1] in punctuation:
		ret = ret[:-1].strip()
	ret = ret.replace(' i ', ' I ')
	return ret

def tsv_reader(dataset, OPTS):

	if OPTS.remove_end_punc:
		collate_fn = lambda ts_list: zip(*[
			[first_lower(l.strip(punctuation)) for l in line.split('\t')] for line in ts_list
		])
	else:
		collate_fn = lambda ts_list: zip(*[
			line.split('\t') for line in ts_list
		])

	loader = DataLoader(
		dataset, batch_size=OPTS.batch_size, drop_last=False,
		collate_fn=collate_fn
	)

	return loader

