task preproc_produce_dict
< because_tr=$encoded_dir@encode_spm[Direction:e2c,Split:train]
< so_tr=$encoded_dir@encode_spm[Direction:e2c,Split:train]
> dict
{

	cat $because_tr/train.spm.effect.seq $so_tr/train.spm.cause.seq > train.input.seq
	cat $because_tr/train.spm.cause.both $so_tr/train.spm.effect.both > train.output.both

	fairseq-preprocess \
		--bpe sentencepiece --joined-dictionary \
		--trainpref "train" \
		--source-lang input.seq \
		--target-lang  output.both \
		--workers $(nproc) \
		--destdir  dummybin

	ln -s `realpath dummybin/dict.input.seq.txt` $dict

}

task training_preproc
< dict=@preproc_produce_dict
< train_dir=$encoded_dir@encode_spm[Split:train]
< dev_dir=$encoded_dir@encode_spm[Split:dev]
:: source_lang=@ target_lang=@
:: source_rep=@ target_rep=@
> outbin
{
	fairseq-preprocess  \
			--bpe sentencepiece --joined-dictionary \
			--source-lang ${source_lang}.${source_rep} \
			--target-lang  ${target_lang}.${target_rep} \
			--trainpref ${train_dir}/train.spm \
			--validpref ${dev_dir}/dev.spm \
			--workers $(nproc) \
			--tgtdict ${dict} \
			--destdir  ${outbin}
}

task eval_preproc
< dict=@preproc_produce_dict
< eval_dir=$encoded_dir@encode_spm[Split:test]
> outbin
:: source_lang=@ target_lang=@
:: source_rep=@ target_rep=@
{
	fairseq-preprocess  \
			--bpe sentencepiece --joined-dictionary \
			--source-lang ${source_lang}.${source_rep} \
			--target-lang  ${target_lang}.${target_rep} \
			--testpref ${eval_dir}/test.spm \
			--workers $(nproc) \
			--tgtdict ${dict} \
			--destdir  ${outbin}

}

task fairseq_train
	< inbin=$outbin@training_preproc
	> out outbin
	:: source_lang=@ source_rep=@ target_lang=@ target_rep=@
	:: device_space=@
	{
#	numtokens=$(( device_space ))

	numtokens=20000

	mkdir $out

	fairseq-train $inbin \
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
		--max-epoch 10 \
		--save-interval 1 \
		--share-all-embeddings | tee -a train.log

	ln -s $inbin $outbin

#			--patience 2 \
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

	ln -s "`dirname ${seq2both}`/out_bin" $out/s2b_bin
	ln -s "`dirname ${seq2prefix}`/out_bin" $out/s2p_bin
	ln -s "`dirname ${b_prefix2seq}`/out_bin" $out/bp2s_bin
	ln -s "`dirname ${b_seq2seq}`/out_bin" $out/bs2s_bin

}



task baseline_generate
	< checkpoint_dir=$out@fairseq_train[SourceRep:seq]
	< spm_model_dir=@train_spm
	< bin=$outbin@eval_preproc
	> results
	:: source_lang=@ target_lang=@
	:: target_rep=@
	:: random_sample=@
	{

	if [ $random_sample == "yes" ]; then
		to_rs="--sampling --sampling-topk  30 --nbest 30 "
	else
		to_rs="--nbest 10"
	fi

	fairseq-generate $bin \
	--path ${checkpoint_dir}/checkpoint_last.pt \
	--beam 30 \
	--max-tokens 8000 \
	--bpe 'sentencepiece' \
	--sentencepiece-vocab ${spm_model_dir}/spm.bpe.model \
	--remove-bpe 'sentencepiece' \
	-s ${source_lang}.${source_rep} -t ${target_lang}.both  \
	--results-path ${results} \
	${to_rs}

	grep "^[^P]" ${results}/generate-test.txt > ${results}/generate_cands.log
	cat ${results}/generate_cands.log | sed 's/^\([STOH].[0-9]\+\).*SEP\$\s\(.*\)$/\1\t\2/' > ${results}/generate_cands_no_pref.log

}

task generate_mmi : ddecode
	< models=$out@ensemble
	< bin=$outbin@eval_preproc
	< spm_model_dir=@train_spm
	> out
	:: bits=$lsh_bits
	:: source_lang=@ target_lang=@
	:: p_inf=(PrefInference: beam bidi) s_inf=(SeqInference: beam bidi)
	:: h_thr=(HammingThreshold: yes=2 no=0)
	:: random_sample=@
	:: debug=@
	{

	if [ $debug == "yes" ]; then
		to_debug="--debug"
	else
		to_debug=""
	fi

	if [ $random_sample == "yes" ]; then
		to_rs="--forward-sequence-sampling"
	else
		to_rs=""
	fi


	PYTHONPATH=$ddecode python ${ddecode}/generate.py $eval_bin \
	--integer-decode \
	--integer-decode-tokens $bits --integer-decode-bits $bits \
	--path $models/s2b_model \
	--backward-prefix-path $models/bp2s_model \
	--backward-prefix-data $models/bp2s_bin \
	--backward-sequence-path $models/bs2s_model \
	--backward-sequence-data $models/bs2s_bin \
	--forward-prefix-path $models/s2p_model \
	--forward-prefix-data $models/s2p_bin \
	--criterion label_smoothed_cross_entropy \
	--prefix-forward-decode beam \
	--prefix-inference ${p_inf} --sequence-inference ${s_inf} \
	--prefix-beam 100 --beam 40 \
	--sequence-bidi-lambda .3 --prefix-bidi-lambda 1000 \
	--prefix-oracle \
	--target-perplexity \
	--max-tokens 2000 \
	--bpe 'sentencepiece' \
	--bucket-distance ${h_thr} \
	--sentencepiece-vocab ${spm_model_dir}/spm.bpe.model \
	--remove-bpe 'sentencepiece' \
	-s ${source_lang}.seq -t ${target_lang}.both  \
	--verbose \
	--order-by prefix \
	--results-path $out \
	$to_rs

	grep "^[^P]" ${out}/generate-test.txt > ${out}/generate_cands.log
	cat ${out}/generate_cands.log | sed 's/^\([STOH].[0-9]\+\).*SEP\$\s\(.*\)$/\1\t\2/' > ${out}/generate_cands_no_pref.log

}



plan fairseq {
	reach ensemble via (DownloadPreprocessed: yes) * (Direction: *) * (Split: *)
}

plan prepro  {
	reach training_preproc via (DownloadPreprocessed: yes) * (Direction: *) * (Split: *)

}






