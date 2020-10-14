#!/bin/bash

# Classification Model

# deepset ode
python train.py --model classification --sub_model odedset --dset modelnet --batch_size 64 --dims 64,256 --categories all \
  --set_hdim 512,512 --T_end 1 --steps 2 --solver rk4 --epochs 120 --test_batch_size 32 --gpu 4 --save ./exp/setode/deepset \
  --set_size 100 --num_blocks 1 --fc_dims 128 --data_dir ./data/ModelNet40_cloud.h5 --seed 123

# transformer ode
python train.py --model classification --sub_model odetrans --dset modelnet --batch_size 64 --dims 64,256 --categories all \
  --set_hdim 256,256 --T_end 1 --steps 2 --solver rk4 --epochs 120 --test_batch_size 32 --gpu 0 --save ./exp/setode/transformer \
  --set_size 100 --num_blocks 1 --fc_dims 128 --data_dir ./data/ModelNet40_cloud.h5 --seed 123