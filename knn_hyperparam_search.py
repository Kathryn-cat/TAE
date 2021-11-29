import os

# knn evaluation command
cmd = '''\
nohup python3 train.py --dataset_name conala \
    --save_dir pretrained_weights/conala \
    --copy_bt \
    --no_encoder_update \
    --monolingual_ratio 0.5 \
    --epochs 80 \
    --just_evaluate \
    --knn \
    --seed 4 \
    --dstore-fp16 \
    --k {} \
    --probe {} \
    --lmbda {} \
    --dstore-size 50044 \
    --dstore-filename datastore/train \
    --indexfile datastore/train_knn.index \
    --no-load-keys'''

# make results file
with open('results.txt', 'w+') as f:
    pass

for probe in [8, 16]:
    for k in [32, 64, 128, 256, 512][::-1]:
        for lmbda in [0.6, 0.7, 0.8, 0.9, 0.95][::-1]:
            try:
                execute = cmd.format(k, probe, lmbda)
                print(execute)
                os.system(execute)
            except KeyboardInterrupt:
                os._exit(0)