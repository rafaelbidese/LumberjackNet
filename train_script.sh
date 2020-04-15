#!/bin/bash

declare -a lr_arr=(0.00001 0.000001 0.0000001)
declare -a btc_arr=(16 32 64)
declare -a cha_arr=(8 16 32 64 128)
declare -a m_ar=(0.1 0.5 0.9)
declare -a fft_arr=(512 1024 2048)
# declare -a p_arr=(0 0.25 0.5 0.75)
declare -a wd_arr=(0.1 0.5 0.9)
declare -a ker_arr=(2 4 6 8)
declare -a mf_arr=(30 40 50)

i=0

for lr in "${lr_arr[@]}"; do
  for btc in "${btc_arr[@]}";do
    for cha in "${cha_arr[@]}";do
      for m in "${m_ar[@]}";do
        for fft in "${fft_arr[@]}";do
          # for p in "${p_arr[@]}";do
            for wd in "${wd_arr[@]}";do
              for k in "${ker_arr[@]}";do
                for mf in "${mf_arr[@]}";do
                ((i++))
                python lumberjack_net.py          \
                              --n_epoch 600         \
                              --lr "$lr"            \
                              --batch_size "$btc"   \
                              --chan1 "$cha"        \
                              --momentum "$m"       \
                              --n_fft "$fft"        \
                              --weight_decay "$wd"  \
                              --kernel "$k"         \
                              --n_mfcc "$mf"        \
                              --exp "exp$i"         
                      
                done
              done
            done
          # done
        done
      done
    done
  done
done
