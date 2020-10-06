#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate pre-processed data with a trained model.
"""

import logging
import math
import os
import sys

from argparse import Namespace
import torch
import numpy as np
from collections import namedtuple
from fairseq import checkpoint_utils, tasks, utils, options
from src.fairseq_options import add_cod3s_arguments
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter
from src.utils import read_sentences, fairseq_cod3_utils as cod3_utils
from copy import copy
import random

random.seed(1234)
import pprint

pp = pprint.PrettyPrinter(width=120, compact=True)
ppr = lambda x: pp.pprint(x)


def main(args):
    print(args)
    args.sentence_avg = 1
    assert args.path is not None, '--path required for generation!'
    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert args.replace_unk is None or args.dataset_impl == 'raw', \
        '--replace-unk requires a raw text dataset (--dataset-impl=raw)'

    if args.results_path is not None:
        os.makedirs(args.results_path, exist_ok=True)
        output_path = os.path.join(args.results_path, 'generate-{}.txt'.format(args.gen_subset))
        with open(output_path, 'w', buffering=1) as h:
            return _main(args, h)
    else:
        return _main(args, sys.stdout)


def _main(args, output_file):
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
        stream=output_file,
    )
    logger = logging.getLogger('fairseq_cli.generate')

    utils.import_user_module(args)

    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 12000
    logger.info(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)

    # Set dictionaries
    try:
        src_dict = getattr(task, 'source_dictionary', None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    # Load ensemble
    logger.info('loading model(s) from {}'.format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        utils.split_paths(args.path),
        arg_overrides=eval(args.model_overrides),
        task=task,
    )

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=args.log_format,
        log_interval=args.log_interval,
        default_log_format=('tqdm' if not args.no_progress_bar else 'none'),
    )

    # Initialize generator
    # gen_timer = StopwatchMeter()
    forward_args = Namespace(**vars(args))
    if args.forward_sequence_sampling:
        forward_args.sampling = True
        forward_args.sampling_topk = forward_args.nbest = forward_args.beam
    generator = task.build_generator(models, forward_args)

    DirectionTask = namedtuple('DirectionTask', 'task, model, generator')
    model_dict = {}
    model_dict['fwd_full'] = DirectionTask(
        task, models[0], generator
    )

    # Load backward models
    if args.prefix_inference == 'bidi':
        logger.info('loading bwd prefix model(s) from {}'.format(args.backward_prefix_path))

        bwd_prefix_args = Namespace(**vars(args))
        bwd_prefix_args.path = args.backward_prefix_path
        bwd_prefix_args.data = args.backward_prefix_data
        bwd_prefix_args.source_lang = args.target_lang.split('.')[0] + '.prefix'
        bwd_prefix_args.target_lang = args.source_lang
        bwd_prefix_task = tasks.setup_task(bwd_prefix_args)
        [bwd_prefix_model], _model_args = checkpoint_utils.load_model_ensemble(
            utils.split_paths(args.backward_prefix_path),
            arg_overrides=eval(args.model_overrides),
            task=bwd_prefix_task,
        )
        model_dict['bwd_prefix'] = DirectionTask(
            bwd_prefix_task, bwd_prefix_model, bwd_prefix_task.build_generator([bwd_prefix_model], bwd_prefix_args))
    else:
        bwd_prefix_model = None

    if args.sequence_inference == 'bidi':
        logger.info('loading bwd sequence model(s) from {}'.format(args.backward_sequence_path))

        bwd_sequence_args = Namespace(**vars(args))
        bwd_sequence_args.path = args.backward_sequence_path
        bwd_sequence_args.data = args.backward_sequence_data
        bwd_sequence_args.source_lang = args.target_lang.split('.')[0] + '.seq'
        bwd_sequence_args.target_lang = args.source_lang
        bwd_sequence_task = tasks.setup_task(bwd_sequence_args)
        [bwd_sequence_model], _model_args = checkpoint_utils.load_model_ensemble(
            utils.split_paths(args.backward_sequence_path),
            arg_overrides=eval(args.model_overrides),
            task=bwd_sequence_task,
        )
        model_dict['bwd_seq'] = DirectionTask(
            bwd_sequence_task, bwd_sequence_model, bwd_sequence_task.build_generator([bwd_sequence_model], bwd_sequence_args))
    else:
        bwd_sequence_model = None

    if args.forward_prefix_path:
        logger.info(f"using {args.forward_prefix_path} for prefix inference")
        fwd_prefix_args = Namespace(**vars(args))
        if args.forward_prefix_sampling:
            fwd_prefix_args.sampling = True
            fwd_prefix_args.sampling_topk = fwd_prefix_args.nbest = fwd_prefix_args.prefix_beam
        fwd_prefix_args.path = args.forward_prefix_path
        fwd_prefix_args.data = args.forward_prefix_data
        fwd_prefix_args.source_lang = args.source_lang
        fwd_prefix_args.target_lang = args.target_lang.split('.')[0] + '.prefix'
        fwd_prefix_task = tasks.setup_task(fwd_prefix_args)
        [fwd_prefix_model], _model_args = checkpoint_utils.load_model_ensemble(
            utils.split_paths(args.forward_prefix_path),
            arg_overrides=eval(args.model_overrides),
            task=fwd_prefix_task,
        )
        model_dict['fwd_prefix'] = DirectionTask(
            fwd_prefix_task, fwd_prefix_model, fwd_prefix_task.build_generator([fwd_prefix_model], fwd_prefix_args))
    else:
        logger.info("no forward prefix model; using full forward model for prefix inference")
        fwd_prefix_model = None
        model_dict['fwd_prefix'] = DirectionTask(task, models[0], task.build_generator(models, args))

    # Optimize ensemble for generation
    for model in models + [m for m in [bwd_prefix_model, bwd_sequence_model, fwd_prefix_model] if m is not None]:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    num_sentences = 0
    has_target = True
    wps_meter = TimeMeter()
    iters = 0
    for sample in progress:
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if 'net_input' not in sample:
            continue
        iters += 1
        if iters > 1 and args.debug: break

        t2s = lambda tensor: tgt_dict.string(tensor, args.remove_bpe, escape_unk=True)

        def numberToBase(n, b, places):
            digits = []
            while n:
                digits.append(int(n % b))
                n //= b
            while len(digits) < places:
                digits = digits + [0]

            return digits[::-1]

        n2b = lambda p: numberToBase(
            p, 2 ** (args.integer_decode_bits / args.integer_decode_tokens),
            args.integer_decode_tokens
        )

        def _num2tensor(n: int):
            int_seq = n2b(n)
            prefix_str = ' '.join([
                ('▁ ' if args.bpe == 'sentencepiece' else '') + f'${x}$'
                for x in int_seq + ['SEP']])
            prefix_tokens = tgt_dict.encode_line(
                prefix_str
            )[:-1].repeat(
                sample['target'].shape[0]
            ).view(
                sample['target'].shape[0], -1
            ).to(sample['target'].device).long()

            return prefix_tokens

        num_signatures = (
            1 if not args.integer_decode else (
                2 ** args.integer_decode_bits
                if not args.prefix_inference else
                args.prefix_top_k
            )
        )
        logger.info(f"total signatures: {num_signatures}")
        logger.info(f"ex signatures: {[n2b(i) for i in range(10)]}")

        args.max_len_b = args.max_len_b + args.integer_decode_tokens
        if args.prefix_inference in ['beam', 'bidi', 'all']:
            if args.prefix_exclude_file:
                exclude = [int(i.split()[0]) for i in read_sentences(args.prefix_exclude_file)][:args.prefix_exclude_n]
                exclude_strings = [
                    ' '.join([*map(lambda x: f"${x}$", n2b(i))]) + ' $SEP$' for i in exclude
                ]
            else:
                exclude = []
                exclude_strings = []

            decode_length = _num2tensor(0).shape[1]
            sample['prefix_length'] = decode_length
            assert model_dict['fwd_prefix'].generator.max_len_a == 0
            model_dict['fwd_prefix'].generator.max_len_b = decode_length
            model_dict['fwd_prefix'].generator.min_len = decode_length
            model_dict['fwd_prefix'].generator.beam_size = args.prefix_beam

            if args.prefix_forward_decode == 'beam':
                if args.verbose: print('inferring prefixes...')
                forward_prefixes = model_dict['fwd_prefix'].task.inference_step(
                    model_dict['fwd_prefix'].generator, [model_dict['fwd_prefix'].model], sample
                )
                for beam in forward_prefixes:
                    for item in beam:
                        item['forward_prefix_score'] = item['score']

                if args.verbose: print('excluding prefixes...')
                if exclude_strings:

                    def _check_excluded(item):
                        inferred_pref = tgt_dict.string(item['tokens'], args.remove_bpe, escape_unk=True)
                        for e in exclude_strings:
                            if inferred_pref.startswith(e):
                                return True
                        return False

                    for beam in forward_prefixes:
                        for item in beam:
                            if _check_excluded(item):
                                item['score'] = -np.inf

                    forward_prefixes = [
                        sorted(beam, key=lambda x: -x['score']) for beam in forward_prefixes
                    ]

            else:
                # forward_prefixes_debug = task.inference_step(generator, models, sample)

                forward_prefixes = [[] for i in range(sample['target'].shape[0])]
                criterion = task.build_criterion(args)

                encoder_out = model_dict['fwd_prefix'].model.encoder(
                    sample['net_input']['src_tokens'],
                    src_lengths=sample['net_input']['src_lengths'],
                    cls_input=None,
                    return_all_hiddens=True,
                )

                from tqdm import tqdm
                for ind in tqdm(random.sample(range(2 ** args.integer_decode_bits), k=500) if args.debug else range(
                        2 ** args.integer_decode_bits), total=2 ** args.integer_decode_bits):
                    if ind in exclude:
                        continue
                    else:
                        prefix_tensor = _num2tensor(ind)
                        prev_output_seq = torch.ones_like(prefix_tensor) * tgt_dict.pad_index
                        prev_output_seq[:, 0] = sample['net_input']['prev_output_tokens'][:, 0]
                        prev_output_seq[:, 1:] = prefix_tensor[:, :-1]

                        sample_copy = copy(sample)
                        sample_copy['target'] = prefix_tensor
                        sample_copy['net_input'] = copy(sample['net_input'])
                        sample_copy['net_input']['prev_output_tokens'] = prev_output_seq
                        # nll, _ ,_ = criterion(models[0], sample_copy, reduce=False)
                        # import pdb;pdb.set_trace()
                        decoder_out = model_dict['fwd_prefix'].model.decoder(
                            prev_output_seq,
                            encoder_out=encoder_out,
                            src_lengths=sample['net_input']['src_lengths'],
                            return_all_hiddens=True,
                        )

                        loss, nll_loss = criterion.compute_loss(
                            model_dict['fwd_prefix'].model, decoder_out, sample_copy, reduce=False)

                        for i, beam in enumerate(forward_prefixes):
                            score = -nll_loss[i].cpu().data.item()
                            beam.append({
                                'tokens': prefix_tensor[i],
                                'score': score,
                                'forward_prefix_score': score
                            })

                for i in range(len(forward_prefixes)):
                    forward_prefixes[i] = sorted(forward_prefixes[i], key=lambda x: -x['score'])[:args.prefix_beam]
            # forward_prefixes = utils.run_all_prefixes(model=models[0], task=task, sample=sample, num2tensor=_num2tensor, args=args)

            if args.prefix_inference == 'bidi':
                # try:
                if args.verbose: print('reranking prefixes...')
                forward_prefixes = cod3_utils.rerank_bidi(
                    forward_prefixes, model_dict=model_dict,
                    direction='bwd_prefix', args=args, sample=sample, tgt_dict=tgt_dict
                )
            # except:
            # 	continue

            if args.bucket_distance:
                if args.verbose: print('hamming thresholding prefixes...')
                import editdistance
                for beam in forward_prefixes:
                    inferred_prefixes = []
                    for item in beam:
                        keep = True
                        cand_prefix = tgt_dict.string(item['tokens'], args.remove_bpe, escape_unk=True)
                        for kept_prefix in inferred_prefixes:
                            if editdistance.eval(kept_prefix, cand_prefix) <= args.bucket_distance:
                                keep = False
                                break

                        if keep:
                            inferred_prefixes.append(cand_prefix)
                        else:
                            item['score'] = -np.inf

                forward_prefixes = [
                    sorted(beam, key=lambda x: -x['score']) for beam in forward_prefixes
                ]

            for beam in forward_prefixes:
                for item in beam:
                    item['prefix_score'] = item['score']

            if args.prefix_oracle:
                prefix_criterion = model_dict['fwd_prefix'].task.build_criterion(args)
                prefix_tensor = (
                        torch.ones_like(sample['target'][:, :decode_length])
                        * sample['target'][:, :decode_length]
                )
                # prefix_tensor[:,-1] = tgt_dict.eos()
                prev_output_seq = torch.ones_like(prefix_tensor) * tgt_dict.pad_index
                prev_output_seq[:, 0] = sample['net_input']['prev_output_tokens'][:, 0]
                prev_output_seq[:, 1:] = prefix_tensor[:, :-1]

                oracle_sample = copy(sample)
                oracle_sample['target'] = prefix_tensor
                oracle_sample['net_input'] = copy(sample['net_input'])
                oracle_sample['net_input']['prev_output_tokens'] = prev_output_seq

                oracle_beams = model_dict['fwd_prefix'].task.inference_step(
                    model_dict['fwd_prefix'].generator, [model_dict['fwd_prefix'].model],
                    oracle_sample, oracle_sample['target'])

                for i, beam_res in enumerate(oracle_beams):
                    # if len(beam_res) != 1: import pdb;pdb.set_trace()
                    beam_res[0]['prefix_score'] = beam_res[0]['forward_prefix_score'] = beam_res[0]['score']
                    beam_res[0]['score'] = np.inf
                    forward_prefixes[i] = beam_res + forward_prefixes[i]

            def _decoded_signatures():
                for ind in range(args.prefix_top_k + (1 if args.prefix_oracle else 0)):
                    yield (
                        [{k: item[ind][k] for k in item[ind].keys() if 'prefix' in k} for item in forward_prefixes],
                        torch.stack([item[ind]['tokens'][:-1] for item in forward_prefixes])
                           )

            signatures = _decoded_signatures()

        else:
            def _all_signatures(num_signatures):
                for p_num in range(num_signatures):
                    yield 1, _num2tensor(p_num)

            signatures = _all_signatures(num_signatures)

        hypos = []

        iter = 0
        for prefix_scores, prefix_tokens in signatures:
            iter += 1
            # prefix_tokens = None
            if not args.integer_decode and args.prefix_size > 0:
                prefix_tokens = sample['target'][:, :args.prefix_size]
            elif args.integer_decode:
                args.prefix_size = prefix_tokens.shape[-1]

            # gen_timer.start()
            if args.verbose: print(f'inferring {iter}th best sequences...')
            beam_out = task.inference_step(generator, models, sample, prefix_tokens)

            # import pdb;pdb.set_trace()
            for score_dict, beam in zip(prefix_scores, beam_out):
                for item in beam:
                    item['forward_sequence_score'] = item['score']
                    item.update(score_dict)

            # if args.sequence_inference == 'beam':
            # 	hypos.append(beam_out)
            if args.sequence_inference == 'bidi':
                if args.verbose: print(f'reranking {iter}th best sequences...')
                beam_out = cod3_utils.rerank_bidi(
                    beam_out, model_dict=model_dict,
                    direction='bwd_seq', args=args, sample=sample, tgt_dict=tgt_dict)

            hypos.append(beam_out)

            num_generated_tokens = sum(len(h[0]['tokens']) for h in hypos[-1])

        if args.target_perplexity:
            full_criterion = model_dict['fwd_full'].task.build_criterion(args)
            prev_output_seq = torch.ones_like(sample['target']) * tgt_dict.pad_index
            prev_output_seq[:, 0] = sample['net_input']['prev_output_tokens'][:, 0]
            prev_output_seq[:, 1:] = sample['target'][:, :-1]

            target_sample = copy(sample)
            target_sample['net_input'] = copy(sample['net_input'])
            target_sample['net_input']['prev_output_tokens'] = prev_output_seq

            target_beams = model_dict['fwd_full'].task.inference_step(
                model_dict['fwd_full'].generator, [model_dict['fwd_full'].model],
                target_sample, target_sample['target'])


            sample['target_ppl'] = [
                (target_beam[0]['positional_scores'], target_beam[0]['score']) for target_beam in target_beams
            ]

        for i, sample_id in enumerate(sample['id'].tolist()):
            has_target = sample['target'] is not None

            # Remove padding
            src_tokens = utils.strip_pad(sample['net_input']['src_tokens'][i, :], tgt_dict.pad())
            target_tokens = None
            if has_target:
                target_tokens = utils.strip_pad(sample['target'][i, :], tgt_dict.pad()).int().cpu()

            # Either retrieve the original sentences or regenerate them from tokens.
            if align_dict is not None:
                src_str = task.dataset(args.gen_subset).src.get_original_text(sample_id)
                target_str = task.dataset(args.gen_subset).tgt.get_original_text(sample_id)
            else:
                if src_dict is not None:
                    src_str = src_dict.string(src_tokens, args.remove_bpe)
                else:
                    src_str = ""
                if has_target:
                    target_str = tgt_dict.string(target_tokens, args.remove_bpe, escape_unk=True)

            if not args.quiet:
                if src_dict is not None:
                    print('S-{}\t{}'.format(sample_id, src_str), file=output_file)
                    # if has_target:
                    if args.target_perplexity:
                        print('T-{}\t{:<.4f}\t{}'.format(
                            sample_id, sample['target_ppl'][i][1] / math.log(2), target_str
                        ), file=output_file)
                    else:
                        print('T-{}\t{}'.format(sample_id, target_str), file=output_file)

            # Process top predictions
            # for prefix_num in range(1 if not args.integer_decode else args.integer_decode):
            # import pdb;pdb.set_trace()
            if args.prefix_inference in ['beam', 'bidi'] and args.order_by == 'prefix':
                out_ordered = [
                    hypos[prefix_num][i][:args.nbest][0]
                    for prefix_num in range(num_signatures + (1 if args.prefix_oracle else 0))
                ]
            else:
                out_ordered = sorted(
                    [
                        hypos[prefix_num][i][:args.nbest][0]
                        for prefix_num in range(num_signatures)
                    ], key=lambda x: -x['score'])

            for j, hypo in enumerate(out_ordered):
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo['tokens'].int().cpu(),
                    src_str=src_str,
                    alignment=hypo['alignment'],
                    align_dict=align_dict,
                    tgt_dict=tgt_dict,
                    remove_bpe=args.remove_bpe,
                )

                if not args.quiet:
                    score = hypo['score'] / math.log(2)  # convert to base 2
                    if j == 0 and args.prefix_oracle:
                        print('O-{}\t{:<.4f}\t{:<.4f}\t{}'.format(
                            sample_id, hypo['forward_prefix_score'] / math.log(2),
                                       hypo['forward_sequence_score'] / math.log(2), hypo_str),
                            file=output_file)
                    else:
                        if args.sequence_inference in ['bidi', 'beam']:
                            # bidi_score = hypo['bidi_score'] / math.log(2)
                            try:
                                print(
                                    'H-{}\t{:<.4f}\t{:<.4f}\t{}'.format(
                                        sample_id, hypo['forward_prefix_score'] / math.log(2),
                                                   hypo['forward_sequence_score'] / math.log(2),
                                        hypo_str),
                                    file=output_file)
                            except:
                                import pdb;pdb.set_trace()
                        else:
                            print('H-{}\t{:<.4f}\t{}'.format(sample_id, score / math.log(2), hypo_str),
                                  file=output_file)
                    print('P-{}\t{}'.format(
                        sample_id,
                        ' '.join(map(
                            lambda x: '{:.4f}'.format(x),
                            # convert from base e to base 2
                            hypo['positional_scores'].div_(math.log(2)).tolist(),
                        ))
                    ), file=output_file)

                    if args.print_alignment:
                        print('A-{}\t{}'.format(
                            sample_id,
                            ' '.join(['{}-{}'.format(src_idx, tgt_idx) for src_idx, tgt_idx in alignment])
                        ), file=output_file)

                    if args.print_step:
                        print('I-{}\t{}'.format(sample_id, hypo['steps']), file=output_file)

                    if getattr(args, 'retain_iter_history', False):
                        for step, h in enumerate(hypo['history']):
                            _, h_str, _ = utils.post_process_prediction(
                                hypo_tokens=h['tokens'].int().cpu(),
                                src_str=src_str,
                                alignment=None,
                                align_dict=None,
                                tgt_dict=tgt_dict,
                                remove_bpe=None,
                            )
                            print('E-{}_{}\t{}'.format(sample_id, step, h_str), file=output_file)

                    # Score only the top hypothesis
                    if has_target and j == 0:
                        if align_dict is not None or args.remove_bpe is not None:
                            # Convert back to tokens for evaluation with unk replacement and/or without BPE
                            target_tokens = tgt_dict.encode_line(target_str, add_if_not_exist=True)

            print(" ", file=output_file)

        wps_meter.update(num_generated_tokens)
        progress.log({'wps': round(wps_meter.avg)})
        num_sentences += sample['nsentences']

    # logger.info('NOTE: hypothesis and token scores are output in base 2')
    # logger.info('Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)'.format(
    # 	num_sentences, gen_timer.n, gen_timer.sum, num_sentences / gen_timer.sum, 1. / gen_timer.avg))
    # if has_target:
    # 	logger.info('Generate {} with beam={}: {}'.format(args.gen_subset, args.beam, scorer.result_string()))



def cli_main():
    parser = options.get_generation_parser()
    add_cod3s_arguments(parser)
    args = options.parse_args_and_arch(parser)
    if args.vanilla:
        from fairseq_cli.generate import main as vanilla_main
        vanilla_main(args)
    else:
        main(args)


if __name__ == '__main__':
    cli_main()
