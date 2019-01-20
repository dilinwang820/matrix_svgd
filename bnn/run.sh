#! /bin/bash

#set -e 

if [ "$#" -ne 2 ];then
    echo "Usage run.sh [svgd|map_kfac|svgd_kfac|mixture_kfac|sgld|psgld] lr"
    exit 1
fi

method=$1
learning_rate=$2

###
##  for small datasets, 
##  SVGD_KFAC, Mixture_KFAC prefer small training epochs 
##  boston=200, wine=50, otherwise, default settings work well
##  learning_rate: 0.005 for svgd, 0.001 for svgd_kfac and mixture_kfac seems work well
###

# protein=50, boston=500, naval=200, kin8nm=200, combined=500, energy=1000, year=10, concrete=500
#declare -A datasets=( [boston]=1000 [concrete]=1000 [energy]=1000 [kin8nm]=200 [naval]=200 [combined]=500 [wine]=1000 [protein]=50 [year]=10 )
declare -A epochs=( [boston]=20 [concrete]=20 [energy]=20 [kin8nm]=20 [naval]=20 [combined]=20 [wine]=20 [protein]=5 [year]=5 )
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


