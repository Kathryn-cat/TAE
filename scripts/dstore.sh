python3 build_dstore.py \
   --dstore_mmap datastore/mined \
   --dstore_size 2090745 \
   --dimension 768 \
   --faiss_index datastore/mined_knn.index \
   --num_keys_to_add_at_a_time 500000 \
   --starting_point 0 \
   --dstore_fp16 \
   --ncentroids 4096

python3 build_dstore.py \
    --dstore_mmap datastore/train \
    --dstore_size 50041 \
    --dimension 768 \
    --faiss_index datastore/train_knn.index \
    --num_keys_to_add_at_a_time 500000 \
    --starting_point 0 \
    --dstore_fp16 \
    --ncentroids 1024
