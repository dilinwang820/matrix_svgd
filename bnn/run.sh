#! /bin/bash

set -e 

if [ "$#" -ne 2 ];then
    echo "Usage run.sh [svgd|map_kfac|svgd_kfac|mixture_kfac|sgld|psgld] lr"
    exit 1
fi

# SGLD, lr = 0.1
# pSGLD, lr = 0.001
# svgd, lr = 0.005
# svgd_kfac, lr = 0.005 
# mixture_kfac, lr = 0.001

method=$1
learning_rate=$2

###
##  for small datasets, 
##  run SVGD_KFAC, Mixture_KFAC with small training epochs 
##  boston=200, wine=50, otherwise, default settings work well
##  e.g., protein=50, boston=500, naval=200, kin8nm=200, combined=500, energy=1000, year=10, concrete=500
##  learning_rate: 0.005 for svgd, 0.005 / 0.001 for svgd_kfac and mixture_kfac seems work well
###

declare -A epochs=( [protein]=5 )
declare -A datasets=( [protein]=50 )

for ds in "${!datasets[@]}"
do
    for((i=1;i<=20;i++))
    do
        if [ $i -le ${epochs[$ds]} ];then
            python trainer.py --method ${method} --dataset $ds --trial $i --n_epoches ${datasets[$ds]}  --learning_rate ${learning_rate}
        fi
    done
done


