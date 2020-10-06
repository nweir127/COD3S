task train_spm : COD3S
	< train_because_dir=(DownloadPreprocessed:
		yes=$because_data@cod3d_data
		no=$out@compute_signatures[Split:train,Direction:e2c])
	< train_so_dir=(DownloadPreprocessed:
		yes=$so_data@cod3d_data
		no=$out@compute_signatures[Split:train,Direction:c2e])
	> spm_model_dir
	:: s_lang=$source_lang
	:: t_lang=$target_lang
	{

	infiles="$train_because_dir/train.cause.both"
	infiles+=",$train_because_dir/train.effect.seq"
	infiles+=",$train_so_dir/train.cause.seq"
	infiles+=",$train_so_dir/train.effect.both"


	mkdir $spm_model_dir
	PYTHONPATH=$COD3S python $COD3S/src/scripts/spm_train.py \
	--input=${infiles} \
	--user_defined_symbols='$0$,$1$,$SEP$' \
	--model_prefix=$spm_model_dir/spm.bpe \
	--vocab_size=10000 \
	--character_coverage=1.0 \
	--model_type=bpe
}


task encode_spm : COD3S
	< in_dir=(DownloadPreprocessed:
		yes=(Split:
			train=(Direction: c2e=$so_data@cod3d_data e2c=$because_data@cod3d_data)
			dev=(Direction: c2e=$so_data@cod3d_data e2c=$because_data@cod3d_data)
			test=(Direction:
				c2e=(EvalData: copa=$cause2effect_copa@cod3d_data causalbank=$cause2effect_cb@cod3d_data)
				e2c=(EvalData: copa=$effect2cause_copa@cod3d_data causalbank=$effect2cause_cb@cod3d_data)
			)
		)
		no=$out@compute_signatures
	)
	< spm_model_dir=@train_spm
	> encoded_dir
	:: s_lang=$source_lang
	:: t_lang=$target_lang
	:: split=@
	{
	mkdir $encoded_dir

	if [ $split == "test" ]; then
		PYTHONPATH=$COD3S python $COD3S/src/scripts/spm_encode.py \
				--inputs $in_dir/$split.$s_lang.seq \
				--outputs $encoded_dir/$split.spm.$s_lang.seq \
				--model ${spm_model_dir}/spm.bpe.model \
				--output_format piece

		PYTHONPATH=$COD3S python $COD3S/src/scripts/spm_encode.py \
				--inputs $in_dir/$split.$t_lang.both \
				--outputs $encoded_dir/$split.spm.$t_lang.both \
				--model ${spm_model_dir}/spm.bpe.model \
				--output_format piece

		f=$encoded_dir/$split.spm.$t_lang.both
		cat $f | sed 's/^.*SEP\$\s\(.*\)$/\1/' > ${f%.both}.seq

	else
		for lang in $s_lang $t_lang; do
			PYTHONPATH=$COD3S python $COD3S/src/scripts/spm_encode.py \
				--inputs $in_dir/$split.$lang.both \
				--outputs $encoded_dir/$split.spm.$lang.both \
				--model ${spm_model_dir}/spm.bpe.model \
				--output_format piece

			f=$encoded_dir/$split.spm.$lang.both
			cat $f | sed 's/^\(.*SEP\$\).*$/\1/' > ${f%.both}.prefix
			cat $f | sed 's/^.*SEP\$\s\(.*\)$/\1/' > ${f%.both}.seq

		done
	fi

}
