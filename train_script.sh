#!/bin/bash

declare -a lr_arr=(0.0001 0.00001 0.000001)
declare -a btc_arr=(1 8 16 32 64)

i=0

for lr in "${lr_arr[@]}"; do
  for btc in "${btc_arr[@]}";do
    python lumberjack_net.py --n_epoch 1 --lr "$lr" --momentum 0.9 --batch_size "$btc" --exp "exp$i"
    ((i++))
  done
done
