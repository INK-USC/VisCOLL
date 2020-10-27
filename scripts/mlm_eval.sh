#!/bin/bash

# usage: ./scripts/mlm_eval.sh <model_name> <model_type (vlbert or lxmert)> <seed>

seed=1
max_seed=3 # modify this

name=${1}

# specify config file here
config=naive

model="vlbert"
if [[ -n "$2" ]]; then
  model=$2
fi

if [[ -n "$3" ]]; then
  seed=$3
fi


lst="1000 2000 4000 6000 8000 10000 12000 14000 16000 18000"

echo "model: ${model}"

while(( ${seed}<=${max_seed} ))
do
  for iteration in ${lst}
  do
    echo "iteration ${iteration} seed ${seed}"
    python test.py --name ${name} --config configs/mlmcaptioning/${config}.yaml --seed ${seed} --epoch 0 --iter ${iteration} --cfg MLMCAPTION.BASE=${model}
    python test.py --name ${name} --config configs/mlmcaptioning/${config}.yaml --seed ${seed} --novel_comps --epoch 0 --iter ${iteration} --cfg MLMCAPTION.BASE=${model}
  done
  echo "test at the end of training"
  python test.py --name ${name} --config configs/mlmcaptioning/${config}.yaml --seed ${seed} --cfg MLMCAPTION.BASE=${model}
  python test.py --name ${name} --config configs/mlmcaptioning/${config}.yaml --seed ${seed} --novel_comps --cfg MLMCAPTION.BASE=${model}
  let seed++
done