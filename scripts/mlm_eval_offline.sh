#!/bin/bash


name=${1}
model="vlbert"
if [[ -n "$2" ]]; then
  model=$2
fi

seed=1
if [[ -n "$3" ]]; then
  seed=$3
fi


./scripts/run_seeds.sh "test.py --name ${name} --config configs/mlmcaptioning/naive.yaml --cfg MLMCAPTION.BASE=${2}" ${seed}
./scripts/run_seeds.sh "test.py --name ${name} --config configs/mlmcaptioning/naive.yaml --novel_comps --cfg MLMCAPTION.BASE=${2}" ${seed}
./scripts/run_seeds.sh "test.py --name ${name} --config configs/mlmcaptioning/naive.yaml --epoch 05 --cfg MLMCAPTION.BASE=${2}" ${seed}
./scripts/run_seeds.sh "test.py --name ${name} --config configs/mlmcaptioning/naive.yaml --novel_comps --epoch 05 --cfg MLMCAPTION.BASE=${2}" ${seed}
