# Uses hyperparams from KNN-MT for now, should probably tune for code gen task

# CUDA_LAUNCH_BLOCKING=1 
python3 train.py --dataset_name conala \
    --save_dir pretrained_weights/conala \
    --copy_bt \
    --no_encoder_update \
    --monolingual_ratio 0.5 \
    --epochs 80 \
    --just_evaluate \
    --knn \
    --seed 4 \
    --dstore-fp16 \
    --k 64 \
    --probe 8 \
    --lmbda 0.8 \
    --dstore-size 2086490 \
    --dstore-filename datastore/mined \
    --indexfile datastore/mined_knn.index \
    --no-load-keys
