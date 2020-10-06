task preproc_produce_dict : fairseq
< because_tr=$encoded_dir@encode_spm[Direction:e2c,Split:train]
< so_tr=$encoded_dir@encode_spm[Direction:e2c,Split:train]
> dict
{

	cat $because_tr/train.spm.effect.seq $so_tr/train.spm.cause.seq > train.input.seq
	cat $because_tr/train.spm.cause.both $so_tr/train.spm.effect.both > train.output.both

	python $fairseq/fairseq_cli/preprocess.py \
		--bpe sentencepiece --joined-dictionary \
		--trainpref "train" \
		--source-lang input.seq \
		--target-lang  output.both \
		--workers $(nproc) \
		--destdir  dummybin

	ln -s `realpath dummybin/dict.input.seq.txt` $dict

}

task training_preproc : fairseq
< dict=@preproc_produce_dict
< train_dir=$encoded_dir@encode_spm[Split:train]
< dev_dir=$encoded_dir@encode_spm[Split:dev]
:: source_lang=@ target_lang=@
:: source_rep=@ target_rep=@
> outbin
{
	PYTHONPATH=$fairseq python $fairseq/fairseq_cli/preprocess.py  \
			--bpe sentencepiece --joined-dictionary \
			--source-lang ${source_lang}.${source_rep} \
			--target-lang  ${target_lang}.${target_rep} \
			--trainpref ${train_dir}/train.spm \
			--validpref ${dev_dir}/dev.spm \
			--workers $(nproc) \
			--tgtdict ${dict} \
			--destdir  ${outbin}
}

task eval_preproc : fairseq
< dict=@preproc_produce_dict
< eval_dir=$encoded_dir@encode_spm[Split:test]
> outbin
:: source_lang=@ target_lang=@
:: source_rep=@ target_rep=@
{
	PYTHONPATH=$fairseq python $fairseq/fairseq_cli/preprocess.py  \
			--bpe sentencepiece --joined-dictionary \
			--source-lang ${source_lang}.${source_rep} \
			--target-lang  ${target_lang}.${target_rep} \
			--testpref ${eval_dir}/test.spm \
			--workers $(nproc) \
			--tgtdict ${dict} \
			--destdir  ${outbin}

}

task fairseq_train : fairseq
	< inbin=$outbin@training_preproc
	> out outbin
	:: source_lang=@ source_rep=@ target_lang=@ target_rep=@
	:: device_space=@ gpu_id=@
	{
	numtokens=$(( device_space  * 4 / 5))
	export CUDA_VISIBLE_DEVICES=$gpu_id

	mkdir $out

	python $fairseq/train.py $inbin \
		--source-lang ${source_lang}.${source_rep} \
		--target-lang ${target_lang}.${target_rep} \
		--save-dir $out \
		--arch transformer_iwslt_de_en \
		--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
		--dropout 0.1 --weight-decay 0 \
		--max-tokens $numtokens \
		--bpe sentencepiece \
		--optimizer adam  --clip-norm 0.1 \
		--adam-betas '(0.9, 0.98)' \
		--lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
		--keep-interval-updates 1 --save-interval-updates 10000  \
		--log-interval 100 \
		--num-workers 0 \
		--max-epoch 8 \
		--patience 2 \
		--save-interval 1 \
		--share-all-embeddings | tee -a train.log

	ln -s $inbin $outbin

}



task ensemble
	< seq2both=(Direction: c2e=$out@fairseq_train[SourceRep:seq,TargetRep:both,Direction:c2e]
						   e2c=$out@fairseq_train[SourceRep:seq,TargetRep:both,Direction:e2c])
	< seq2prefix=(Direction: c2e=$out@fairseq_train[SourceRep:seq,TargetRep:prefix,Direction:c2e]
	 						 e2c=$out@fairseq_train[SourceRep:seq,TargetRep:prefix,Direction:e2c])
	< b_prefix2seq=(Direction: c2e=$out@fairseq_train[SourceRep:prefix,TargetRep:seq,Direction:e2c]
							   e2c=$out@fairseq_train[SourceRep:prefix,TargetRep:seq,Direction:c2e])
	< b_seq2seq=(Direction: c2e=$out@fairseq_train[SourceRep:seq,TargetRep:seq,Direction:e2c]
	 						e2c=$out@fairseq_train[SourceRep:seq,TargetRep:seq,Direction:c2e])
	> out {
	mkdir $out
	ln -s $seq2both/checkpoint_last.pt $out/s2b_model
	ln -s $seq2prefix/checkpoint_last.pt $out/s2p_model
	ln -s $b_prefix2seq/checkpoint_last.pt $out/bp2s_model
	ln -s $b_seq2seq/checkpoint_last.pt $out/bs2s_model

	ln -s "`dirname ${seq2both}`/outbin" $out/s2b_bin
	ln -s "`dirname ${seq2prefix}`/outbin" $out/s2p_bin
	ln -s "`dirname ${b_prefix2seq}`/outbin" $out/bp2s_bin
	ln -s "`dirname ${b_seq2seq}`/outbin" $out/bs2s_bin

}

plan train {
	reach ensemble via (DownloadPreprocessed: yes) * (Direction: e2c) * (Split: *)
	reach ensemble via (DownloadPreprocessed: yes) * (Direction: c2e) * (Split: *)
}











