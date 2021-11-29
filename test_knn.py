# Script for debugging knnlm model
import os
from utils import *
from torch.utils.data import DataLoader 
from dataset_preprocessing.conala import Conala
from knnlm import KNN_Dstore
from model import *
import pickle

parser = make_parser()
args = get_args(parser)

args.save_dir = 'pretrained_weights/conala'
args.dstore_fp16 = True
args.k = 64
args.probe = 8
args.lmbda = 0.8
args.dstore_size = 50041
args.dstore_filename = 'datastore/train'
args.indexfile = 'datastore/train_knn.index'
args.no_load_keys = True
args.dataset_name = 'conala'
args.pointer_network = False

dstore = KNN_Dstore(args)
model = KNNModel(dstore, 'bert-base-uncased', args)
model.to(args.device)
model.load_state_dict(torch.load(os.path.join(args.save_dir, 'conala_weights.pth')))
model.eval()

def load_dataset(args, tokenizer):
    splits = ['train', 'dev', 'test']
    datasets = []
    for split in splits:
        dataset = Conala(args.dataset_name, split, tokenizer, args)
        datasets.append(dataset)
    return (*datasets,) if len(datasets) > 1 else dataset

train_dataset, valid_dataset, test_dataset = load_dataset(args, model.tokenizer)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
    num_workers=args.num_workers, pin_memory=True, collate_fn=preprocess_batch)

keys = np.memmap(f'datastore/train_keys.npy', dtype=np.float16, mode='r',
                        shape=(50014, 768))
vals = np.memmap(f'datastore/train_vals.npy', dtype=np.int32, mode='r',
                        shape=(50014,))

data = next(iter(train_loader))
input_ids = data['target']['input_ids']

logits, *_, prediction, generation_prediction = model(data)

query = generation_prediction[:, 0].unsqueeze(1)

knn_scores, (knns, probs, indices) = model.get_knn_scores_per_step(query)

with open('datastore/mined_kv_pairs.p', 'rb') as f:
    kv_pairs = pickle.load(f)
