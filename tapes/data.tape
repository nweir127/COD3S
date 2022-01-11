task data
	> because_data so_data
	> cause2effect_cb effect2cause_cb cause2effect_copa effect2cause_copa
	{
	wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1sPNrR-nrdBWlovJeSgbiqr1anruY0Yeh' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1sPNrR-nrdBWlovJeSgbiqr1anruY0Yeh"  -O COD3S_train-eval_data.zip && rm -rf /tmp/cookies.txt
	unzip COD3S_train-eval_data.zip

	echo "If you use this dataset, please cite https://www.ijcai.org/Proceedings/2020/0502.pdf (Li et al. 2020)"

	# make visible to ducttape
	ln -s `realpath cause2effect/cb` $cause2effect_cb
	ln -s `realpath cause2effect/copa` $cause2effect_copa
	ln -s `realpath effect2cause/cb` $effect2cause_cb
	ln -s `realpath effect2cause/copa` $effect2cause_copa

}


plan download_data {
	reach data
}



task cod3d_data
	> because_data so_data
	> cause2effect_cb effect2cause_cb cause2effect_copa effect2cause_copa
	{
	file_id="1m8t66Tj6hlu0PNYnf8MmV2prMlEAhk2k"
	wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=${file_id}" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${file_id}"  -O cod3d_causalbank.zip && rm -rf /tmp/cookies.txt
	unzip cod3d_causalbank.zip

	# make visible to ducttape
	ln -s `realpath so` $so_data
	ln -s `realpath because` $because_data
	ln -s `realpath e2c/cb` $effect2cause_cb
	ln -s `realpath e2c/copa` $effect2cause_copa
	ln -s `realpath c2e/cb` $cause2effect_cb
	ln -s `realpath c2e/copa` $cause2effect_copa

}

plan download_cod3d_data {
	reach cod3d_data
}


task human_eval_data > cause2effect effect2cause {
	file_id="1NPiestgnFRvCIFwznIlKtzAKfity7aYX"
	wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=${file_id}" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${file_id}"  -O human_eval_outputs.zip && rm -rf /tmp/cookies.txt
	unzip human_eval_outputs.zip
}

plan download_human_eval {
	reach human_eval_data
}


task pretrained_models > out {
	# file_id="1jQI993bm1orzh3oywWseWYTe8T3fknfQ"
	# wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=${file_id}" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${file_id}"  -O pretrained_cod3s_models.zip && rm -rf /tmp/cookies.txt
	wget https://nlp.jhu.edu/causalbank/pretrained_cod3s_models.zip
	unzip pretrained_cod3s_models.zip
	ln -s exp/fairseq_train $out
}

task gather_pretrained_model < in=$out@pretrained_models > out outbin
:: direction=(Direction: c2e e2c)
:: source_rep=@ target_rep=@
{
	ln -s $in/Direction.${direction}+DownloadPreprocessed.yes+SourceRep.${source_rep}+TargetRep.${target_rep}/out $out
	ln -s $in/Direction.${direction}+DownloadPreprocessed.yes+SourceRep.${source_rep}+TargetRep.${target_rep}/outbin $outbin


}

task pretrained_ensemble
	< seq2both=(Direction: c2e=$out@gather_pretrained_model[SourceRep:seq,TargetRep:both,Direction:c2e]
						   e2c=$out@gather_pretrained_model[SourceRep:seq,TargetRep:both,Direction:e2c])
	< seq2prefix=(Direction: c2e=$out@gather_pretrained_model[SourceRep:seq,TargetRep:prefix,Direction:c2e]
	 						 e2c=$out@gather_pretrained_model[SourceRep:seq,TargetRep:prefix,Direction:e2c])
	< b_prefix2seq=(Direction: c2e=$out@gather_pretrained_model[SourceRep:prefix,TargetRep:seq,Direction:e2c]
							   e2c=$out@gather_pretrained_model[SourceRep:prefix,TargetRep:seq,Direction:c2e])
	< b_seq2seq=(Direction: c2e=$out@gather_pretrained_model[SourceRep:seq,TargetRep:seq,Direction:e2c]
	 						e2c=$out@gather_pretrained_model[SourceRep:seq,TargetRep:seq,Direction:c2e])
	> out {
	mkdir $out
	ln -s $seq2both/checkpoint_last.pt $out/s2b_model
	ln -s $seq2prefix/checkpoint_last.pt $out/s2p_model
	ln -s $b_prefix2seq/checkpoint_last.pt $out/bp2s_model
	ln -s $b_seq2seq/checkpoint_last.pt $out/bs2s_model

	# hack to not duplicate parameters
	ln -s "`dirname ${seq2both}`/outbin" $out/s2b_bin
	ln -s "`dirname ${seq2prefix}`/outbin" $out/s2p_bin
	ln -s "`dirname ${b_prefix2seq}`/outbin" $out/bp2s_bin
	ln -s "`dirname ${b_seq2seq}`/outbin" $out/bs2s_bin

}

plan download_pretrained_models {
	reach pretrained_models
}
