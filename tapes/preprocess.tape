


func compute_signatures : COD3S
	< data_dir
	> out
	:: source_lang target_lang split lsh_bits
	:: gpu_id device_space distributed
	:: debug {

	if [ $distributed == no ]; then
		gpu_id=`echo $gpu_id | cut -f 1 -d ','`
		num_gpus=1
	else
		num_gpus=$(echo $gpu_id | tr -cd , | wc -c)
		num_gpus=$((num_gpus + 1))
	fi

	jobcounter=1

	for gpu_id_0 in $(echo $gpu_id | sed "s/,/ /g"); do

		params="--input ${data_dir}/${split}.tsv --output_dir worker_${gpu_id_0} --split ${split}"
		params+=" --lsh_dim ${lsh_bits}"
		params+=" --source_lang ${source_lang} --target_lang ${target_lang}"
		params+=" --source_rep both,seq --target_rep both,seq"
		params+=" --debug $debug --gpuid ${gpu_id_0}"
		batch_size=$(($device_space / 11 ))
		params+=" --batch_size ${batch_size}"

		params+=" --local_rank $jobcounter --num_thread $num_gpus"

		PYTHONPATH=$COD3S python $COD3S/src/scripts/compute_signatures.py ${params} &
		pids[${jobcounter}]=$!
		jobcounter=$((jobcounter + 1))

	done

	for pid in ${pids[*]}; do
    	wait $pid
	done

	mkdir -p $out
	for worker_out in worker_*; do
			for f in `ls $worker_out`; do
			cat $worker_out/$f >> $out/$f
		done
	done

}



task compute_train_signatures calls compute_signatures : COD3S
	< data_dir=(Direction: c2e=$so_data@data e2c=$because_data@data)
	> out
	:: source_lang=@ target_lang=@ split=@ lsh_bits=@
	:: distributed=(Split: train="yes" dev="no" test="no")
	:: gpu_id=$gpu_id
	:: device_space=@
	:: debug=(Debug: no=0 yes=1)


task compute_eval_signatures calls compute_signatures : COD3S
	< data_dir=(EvalData:
		copa=(Direction: c2e=$cause2effect_copa@data e2c=$effect2cause_copa@data)
		causalbank=(Direction: c2e=$cause2effect_cb@data e2c=$effect2cause_cb@data))
	> out
	:: source_lang=@ target_lang=@ split=@ lsh_bits=@
	:: gpu_id=@ device_space=@
	:: distributed="no"
	:: debug=(Debug: no=0 yes=1)

task compute_signatures
	< out_dir=(Split: train=$out@compute_train_signatures
		   		  dev=$out@compute_train_signatures
		   		  test=$out@compute_eval_signatures)
	> out { ln -s $out_dir $out }


plan compute_cod3s {
	reach compute_signatures via (Direction: *) * (Split: *) * (EvalData: *) * (Debug: yes)
}
