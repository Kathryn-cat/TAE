python3 store_embeds.py \
    --dataset_name conala \
    --data_type train \
    --pretrained_path pretrained_weights/conala/conala_weights.pth

python3 store_embeds.py \
    --dataset_name conala \
    --data_type mined \
    --pretrained_path pretrained_weights/conala/conala_weights.pth

python3 store_embeds.py \
    --dataset_name conala \
    --data_type mined \
    --save_kv_pairs \
    --pretrained_path pretrained_weights/conala/conala_weights.pth
