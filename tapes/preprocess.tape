


func compute_signatures : COD3S
	< data_dir
	> out
	:: source_lang target_lang split lsh_bits
	:: gpu_id device_space distributed num_gpus
	:: debug {

	# script can only use one device
	gpu_id_0=`echo $gpu_id | cut -f 1 -d ','`

	params="--input ${data_dir}/${split}.tsv --output_dir $out --split ${split}"
	params+=" --lsh_dim ${lsh_bits}"
	params+=" --source_lang ${source_lang} --target_lang ${target_lang}"
	params+=" --source_rep both,seq --target_rep both,seq"
	params+=" --debug $debug --gpuid ${gpu_id_0}"
	batch_size=$(($device_space / 11 ))
	params+=" --batch_size ${batch_size}"

	if [ $distributed == "yes" ]; then
		params+=" --local_rank $((gpu_id_0 + 1)) --num_thread $num_gpus"
	fi

	mkdir -p $out
	PYTHONPATH=$COD3S python $COD3S/src/scripts/compute_signatures.py ${params}

}



task _compute_train_signatures calls compute_signatures : COD3S
	< data_dir=(InferenceDirection: c2e=$so_data@data e2c=$because_data@data)
	> out
	:: source_lang=@ target_lang=@ split=@ lsh_bits=@
	:: distributed=(DistributedPrepro: yes no)
	:: gpu_id=(DistributedPrepro: yes=$dist_gpu_id no=$gpu_id) num_gpus=@
	:: device_space=@
	:: debug=(Debug: no=0 yes=1)

task compute_train_signatures
	< out=$out@_compute_train_signatures[DistGPUID:*]
	> joined_out
	{
		mkdir $joined_out
		for worker_out in $out; do
			for f in `ls $worker_out`; do
			cat $worker_out/$f >> $joined_out/$f
			done
		done

}

task compute_dev_signatures calls compute_signatures : COD3S
	< data_dir=(InferenceDirection: c2e=$so_data@data e2c=$because_data@data)
	> out
	:: source_lang=@ target_lang=@ split=@ lsh_bits=@
	:: distributed="no"
	:: gpu_id=@ device_space=@
	:: debug=(Debug: no=0 yes=1)

task compute_eval_signatures calls compute_signatures : COD3S
	< data_dir=(EvalData:
		copa=(InferenceDirection: c2e=$cause2effect_copa@data e2c=$effect2cause_copa@data)
		causalbank=(InferenceDirection: c2e=$cause2effect_cb@data e2c=$effect2cause_cb@data))
	> out
	:: source_lang=@ target_lang=@ split=@ lsh_bits=@
	:: gpu_id=@ device_space=@
	:: distributed="no"
	:: debug=(Debug: no=0 yes=1)

plan tr_e2c {
	reach compute_train_signatures via (InferenceDirection: e2c) * (Split: train) * (Debug: no)
#	reach compute_train_signatures via (InferenceDirection: c2e) * (Split: train) * (Debug: yes) * (DistGPUID: 4 5 6 7)
}

plan tr_c2e {
	reach compute_train_signatures via (InferenceDirection: c2e) * (Split: train) * (Debug: no)
}
#plan compute_cod3s {
#	reach compute_train_signatures via (InferenceDirection: *) * (Split: train) * (Debug: yes)
#	reach compute_dev_signatures via (InferenceDirection: *) * (Split: dev) * (EvalData: *) * (Debug: yes)
#	reach compute_eval_signatures via (InferenceDirection: *) * (Split: test) * (EvalData: *) * (Debug: yes)
#}