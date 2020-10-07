task baseline_generate : COD3S fairseq
	< checkpoint_dir=(Pretrained: no=$out@fairseq_train[SourceRep:seq] yes=$out@gather_pretrained_model[SourceRep:seq])
	< spm_model_dir=@train_spm
	< bin=$outbin@eval_preproc
	> results
	:: source_lang=@ target_lang=@
	:: source_rep="seq" target_rep=@
	:: random_sample=@
	{

	if [ $random_sample == "yes" ]; then
		to_rs="--sampling --sampling-topk  30 --nbest 30 "
	else
		to_rs="--nbest 10"
	fi

	PYTHONPATH=$COD3S:$fairseq python ${COD3S}/src/scripts/mmi_generate.py $bin \
	--path ${checkpoint_dir}/checkpoint_last.pt \
	--vanilla \
	--beam 30 \
	--max-tokens 8000 \
	--remove-bpe 'sentencepiece' \
	-s ${source_lang}.${source_rep} -t ${target_lang}.${target_rep}  \
	--results-path ${results} \
	${to_rs}

	grep "^[^PD]" ${results}/generate-test.txt > ${results}/generate_cands.log
	cat ${results}/generate_cands.log | sed 's/^\([STOH].[0-9]\+\).*SEP\$\s\(.*\)$/\1\t\2/' > ${results}/generate_cands_no_pref.log

}

task generate_mmi : COD3S fairseq
	< models=(Pretrained: no=$out@ensemble yes=$out@pretrained_ensemble)
	< eval_bin=$outbin@eval_preproc
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


	PYTHONPATH=$COD3S:$fairseq python ${COD3S}/src/scripts/mmi_generate.py $eval_bin \
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
	--sentencepiece-model ${spm_model_dir}/spm.bpe.model \
	--remove-bpe 'sentencepiece' \
	-s ${source_lang}.seq -t ${target_lang}.both  \
	--verbose \
	--order-by prefix \
	--results-path $out \
	$to_rs \
	${to_debug}

	grep "^[^P]" ${out}/generate-test.txt > ${out}/generate_cands.log
	cat ${out}/generate_cands.log | sed 's/^\([STOH].[0-9]\+\).*SEP\$\s\(.*\)$/\1\t\2/' > ${out}/generate_cands_no_pref.log

}

plan generate_all {
# seq2seq_baseline {
reach baseline_generate via (DownloadPreprocessed: yes) * (Direction: *) * (EvalData: *)
* (TargetRep:seq) * (RandomSampling: no)

# seq2both_baseline {
reach baseline_generate via (DownloadPreprocessed: yes) * (Direction: *) * (EvalData: *)
* (TargetRep:both) * (RandomSampling: no)

# s2s_rs {
reach baseline_generate via (DownloadPreprocessed: yes) * (Direction: *) * (EvalData: *)
* (TargetRep:seq) * (RandomSampling: yes)

# beam_beam {
reach generate_mmi via (DownloadPreprocessed: yes) * (Direction: *) * (EvalData: *)
	* (PrefInference: beam) * (SeqInference: beam) * (RandomSampling: no) * (HammingThreshold: yes)

# beam_beam_rs {
reach generate_mmi via (DownloadPreprocessed: yes) * (Direction: *) * (EvalData: *)
	* (PrefInference: beam) * (SeqInference: beam) * (RandomSampling: yes) * (HammingThreshold: yes)

# beam_bidi {
reach generate_mmi via (DownloadPreprocessed: yes) * (Direction: *) * (EvalData: *)
	* (PrefInference: beam) * (SeqInference: bidi) * (RandomSampling: no) * (HammingThreshold: yes)


# beam_bidi_no_div {
reach generate_mmi via (DownloadPreprocessed: yes) * (Direction: *) * (EvalData: *)
	* (PrefInference: beam) * (SeqInference: bidi) * (RandomSampling: no) * (HammingThreshold: no)

# beam_bidi_rs {
reach generate_mmi via (DownloadPreprocessed: yes) * (Direction: *) * (EvalData: *)
	* (PrefInference: beam) * (SeqInference: bidi) * (RandomSampling: yes) * (HammingThreshold: yes)

# bidi_bidi {
reach generate_mmi via (DownloadPreprocessed: yes) * (Direction: *) * (EvalData: *)
	* (PrefInference: bidi) * (SeqInference: bidi) * (RandomSampling: no) * (HammingThreshold: yes)

# bidi_bidi_no_div {
reach generate_mmi via (DownloadPreprocessed: yes) * (Direction: *) * (EvalData: *)
	* (PrefInference: bidi) * (SeqInference: bidi) * (RandomSampling: no) * (HammingThreshold: no)

# bidi_bidi_rs {
reach generate_mmi via (DownloadPreprocessed: yes) * (Direction: *) * (EvalData: *)
	* (PrefInference: bidi) * (SeqInference: bidi) * (RandomSampling: yes) * (HammingThreshold: yes)

# bidi_bidi_rs_no_div {
reach generate_mmi via (DownloadPreprocessed: yes) * (Direction: *) * (EvalData: *)
	* (PrefInference: bidi) * (SeqInference: bidi) * (RandomSampling: yes) * (HammingThreshold: no)
}

