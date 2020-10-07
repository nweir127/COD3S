package COD3S :: .versioner=disk .path="/srv/local1/nweir/COD3S" {}

package fairseq
    :: .versioner=git .repo="https://github.com/nweir127/fairseq-COD3S" .ref=HEAD {
  pip install --editable .
}

global {
    # specify visible device id(s)
	gpu_id=0,1


	# MB on gpu, used for batch sizes
	device_space="24220"

	debug=(Debug: no yes)

	# housekeeping
	ducttape_output="exp"
	ducttape_experimental_packages=true
	ducttape_experimental_submitters=true
	ducttape_experimental_imports=true
	ducttape_experimental_multiproc=true

	.submitter="shell"
	.action_flags=""
	.resource_flags=""
}

import "params.tape"

import "data.tape"

import "preprocess.tape"

import 'spm.tape'
import "training.tape"

import 'inference.tape'


plan generate_cod3s {
reach generate_mmi via (DownloadPreprocessed: yes) * (Direction: *) * (EvalData: copa)
	* (PrefInference: bidi) * (SeqInference: bidi) * (RandomSampling: yes) * (HammingThreshold: yes) * (Pretrained: yes)

}

plan generate_seq2seq {
reach baseline_generate via (DownloadPreprocessed: yes) * (Direction: *) * (EvalData: copa)
	* (TargetRep:seq) * (RandomSampling: no) * (Pretrained: yes)
}
