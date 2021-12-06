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
    --knn_temp {} \
    --dstore-size 2090745 \
    --dstore-filename datastore/mined \
    --indexfile datastore/mined_knn.index \
    --no-load-keys'''

# make results file
with open('results.txt', 'w+') as f:
    pass

for temp in [1000]:
    for probe in [32]:
        for k in [32, 64, 128, 256]:
            for lmbda in [0.01, 0.05, 0.1, 0.15, 0.20, 0.25]:
                execute = cmd.format(k, probe, lmbda, temp)
                print(execute)
                os.system(execute)