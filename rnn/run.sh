#! /bin/bash

set -e

for method in SVGD_KFAC MIXTURE_KFAC SVGD pSGLD SGLD
do
    for dataset in mr cr subj mpqa
    do
        THEANO_FLAGS="floatX=float32,device=cuda0,force_device=True" python acl_sentence_classification.py --dataset ${dataset} --n_p 10 --method ${method}  
    done
done
