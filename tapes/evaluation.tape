func evaluate_fn : COD3S
< results_dir
> auto_eval {
	PYTHONPATH=$COD3S python $COD3S/src/scripts/diversity_evaluation.py \
	--results_dir=$results_dir
#	--results_dir $COD3S/exp
}

task evaluate calls evaluate_fn : COD3S
< results_dir=""
> auto_eval

plan eval {
	reach evaluate
}