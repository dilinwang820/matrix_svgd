#! /bin/bash

#set -e

dataset="covtype"

n_epoches=2
batch_size=256

for((trial=1;trial<=20;trial++))
do
    for method in mixture_kfac SGLD pSGLD svgd svgd_kfac
    do
        # we search the best learning rate for all baseline models
        for lr in 1e-3 5e-3 1e-2 5e-2 1e-1 #5e-1 1e0 5e0
        do
            logfile="${dataset}_logs/${method}.${lr}.${trial}.log"
            CUDA_VISIBLE_DEVICES=1 python trainer.py --method ${method} --learning_rate ${lr} --trial ${trial} --dataset ${dataset} --n_epoches ${n_epoches} --batch_size ${batch_size} > ${logfile} &
        done
        wait
    done
done
