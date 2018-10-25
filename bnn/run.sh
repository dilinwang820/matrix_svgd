#! /bin/bash

#set -e 

if [ "$#" -ne 1 ];then
    echo "Usage run.sh [svgd|map_kfac|svgd_kfac|mixture_kfac]"
    exit 1
fi

method=$1
#declare -A datasets=( [boston]=200 [concrete]=200 [energy]=200 [kin8nm]=20 [naval]=20 [combined]=40 [wine]=40 [yacht]=200 [protein]=20 [year]=2 )
#declare -A epochs=( [boston]=20 [concrete]=20 [energy]=20 [kin8nm]=20 [naval]=20 [combined]=20 [wine]=20 [yacht]=20 [protein]=5 [year]=5 )

declare -A datasets=( [boston]=200 )
declare -A epochs=( [boston]=20 )

for ds in "${!datasets[@]}"
do
    for((i=1;i<=20;i++))
    do
        if [ $i -le ${epochs[$ds]} ];then
            out=1
            until [ $out -eq 0 ]
            do
                python trainer.py --method ${method} --dataset $ds --trial $i --n_epoches ${datasets[$ds]} 
                out=$?  ## error code
            done
        fi
    done
done


