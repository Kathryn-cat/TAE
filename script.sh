python3 store_embeds.py --pretrained_path pretrained_weights/conala_model_combined_training=False_seed=4_trns_back=False_use_backtr=False_lmd=1_cp_bt=True_add_no=False_no_en_upd=True_ratio=0.5_ext_li=True_ext_cp_li=True_cp_att=True_EMA=T_rnd_enc=F_de_lr=7.5e-05_mmp=0.1_saug=F_dums=F_dumQ=F_rsr=F_fc=F.pth --dataset_name conala

python3 store_embeds.py --pretrained_path pretrained_weights/conala_model_combined_training=False_seed=4_trns_back=False_use_backtr=False_lmd=1_cp_bt=True_add_no=False_no_en_upd=True_ratio=0.5_ext_li=True_ext_cp_li=True_cp_att=True_EMA=T_rnd_enc=F_de_lr=7.5e-05_mmp=0.1_saug=F_dums=F_dumQ=F_rsr=F_fc=F.pth --dataset_name conala --data_type mined

python3 build_dstore.py --dstore_mmap datastore/train --dstore_size 50044 --dimension 768 --faiss_index datastore/train_knn.index --num_keys_to_add_at_a_time 500000 --starting_point 0 --dstore_fp16

ython3 build_dstore.py --dstore_mmap datastore/mined --dstore_size 2090745 --dimension 768 --faiss_index datastore/mined_knn.index --num_keys_to_add_at_a_time 500000 --starting_point 0 --dstore_fp16
