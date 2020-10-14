
# spatial mnist
python flow.py --num_blocks=12 --dim=128,128,128 --data=spatial_mnist --chn_size=2 --set_size=50 --batch_size=1024 \
--epochs=1000 --gpu=3 --data_dir=./data --exp_dir=./exp/flow/spatial_mnist

# airplane
python flow.py --num_blocks=12 --dim=128,128,128 --data=modelnet --category=airplane --chn_size=3 --set_size=512 --batch_size=100 \
--epochs=1000 --gpu=3 --data_dir=./data --exp_dir=./exp/flow/airplane

# chair
python flow.py --num_blocks=12 --dim=128,128,128 --data=modelnet --category=chair --chn_size=3 --set_size=512 --batch_size=100 \
--epochs=1000 --gpu=3 --data_dir=./data --exp_dir=./exp/flow/chair