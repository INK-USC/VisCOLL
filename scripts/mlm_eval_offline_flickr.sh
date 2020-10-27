#!/bin/bash


name=${1}
model="vlbert"
if [[ -n "$2" ]]; then
  model=$2
fi


./scripts/run_seeds.sh "test.py --name ${name} --config configs/mlmcaptioning/naive_flickr.yaml --cfg MLMCAPTION.BASE=${2}"
./scripts/run_seeds.sh "test.py --name ${name} --config configs/mlmcaptioning/naive_flickr.yaml --epoch 05 --cfg MLMCAPTION.BASE=${2}"
