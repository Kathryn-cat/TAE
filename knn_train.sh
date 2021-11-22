# Uses hyperparams from KNN-MT for now, should probably tune for code gen task

python3 train.py --dataset_name conala \
    --save_dir pretrained_weights/conala \
    --copy_bt \
    --no_encoder_update \
    --monolingual_ratio 0.5 \
    --epochs 80 \
    --just_evaluate \
    --seed 4 \
    --knn \
    --dstore-fp16 \
    --k 64 \ # TODO: tune this
    --probe 8 \ # TODO: tune this
    --lmbda 0.8 \ # TODO: tune this
    --dstore-size {size} \ # TODO: fill in this with size outputted by store_embeds.py
    --dstore_filename {datastore/train} \ # train datastore for now, should try mined as well
    --indexfile {wherever faiss index was saved} \ # TODO: fill in with location of faiss index
    --no-load-keys