global {
	split=(Split: train dev test)
	eval_data=(EvalData: copa causalbank)
	source_lang=(Direction: c2e="cause" e2c="effect")
	target_lang=(Direction: c2e="effect" e2c="cause")
	lsh_bits=(Bits: 16)
	download_cod3d=(DownloadPreprocessed: yes no)
	source_rep=(SourceRep: seq prefix)
	target_rep=(TargetRep: both prefix seq)
	random_sample=(RandomSampling: no yes)
}