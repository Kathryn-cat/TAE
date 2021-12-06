'''
Stores each context (i.e. intent/previous tokens) as a fixed-size embedding

Run with:
python3 store_embeds.py \
    --dataset_name conala \
    --pretrained_path {path/to/pretrained/model}
    --data_type ['train', 'mined']

Afterward, run build_dstore.py with appropriate args to generate datastore.
'''

import os
from tqdm import tqdm
import pickle

import torch
from torch.utils.data import DataLoader

import numpy as np
from dataset_preprocessing.conala import Conala
from utils import make_parser, get_args, preprocess_batch
from model import Model

parser = make_parser()
parser.add_argument('--pretrained_path', type=str, help='Location of pretrained model')
parser.add_argument('--data_type', type=str, 
    choices=['train', 'mined', 'csn'], default='train', 
    help='Type of data used in the store (each is a superset of the previous)')
args = get_args(parser)

print(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Model('bert-base-uncased', args)
model.to(device)
model.device = device
model.load_state_dict(torch.load(args.pretrained_path))
model.eval()

data_types = ['train', 'mined', 'csn']
data_types = data_types[:data_types.index(args.data_type)+1]

datasets = []
if 'train' in data_types:
    datasets.append(Conala('conala', 'train', model.tokenizer, args, monolingual=False))
if 'mined' in data_types:
    datasets.append(Conala('conala', 'train', model.tokenizer, args, monolingual=True))
if 'csn' in data_types:
    raise NotImplementedError # need to get code-only data first

full_dataset = datasets[0]
for i in range(1, len(datasets)):
    full_dataset.data.extend(datasets[i].data)

loader = DataLoader(full_dataset, batch_size=args.test_batch_size, shuffle=False,
                    num_workers=0, pin_memory=False, collate_fn=preprocess_batch)

dstore_size = 0
# for i in range(len(full_dataset)):
#     example = full_dataset[i]
#     dstore_size += (len(example['snippet']['input_ids']) - 1)

with torch.no_grad():
    for data in tqdm(loader):
        lengths = data['target']['attention_mask'].sum(dim=1) - 1
        dstore_size += lengths.sum()

key_size = model.encoder.config.hidden_size * 2
    
print('Total # of target tokens:', dstore_size)
print('Size of each key:', )

if not os.path.isdir('datastore'):
    os.mkdir('datastore')

dstore_keys = np.memmap(f'datastore/{args.data_type}_keys.npy', dtype=np.float16, mode='w+',
                        shape=(dstore_size, key_size))
dstore_vals = np.memmap(f'datastore/{args.data_type}_vals.npy', dtype=np.int32, mode='w+',
                        shape=(dstore_size, 1))

kv_pairs = []

with torch.no_grad():
    offset = 0
    for i, data in enumerate(tqdm(loader)):
        lengths = data['target']['attention_mask'].sum(dim=1)
        *_, prediction, last_ffn = model(data, ret_last_ffn=True)
        input_ids = data['target']['input_ids']
        first = i == 0
        for embed, length, ids in zip(last_ffn, lengths, input_ids):
            actual_length = length-1
            for i in range(actual_length):
                context = model.tokenizer.decode(ids[:i+1].cpu().tolist())
                target = model.tokenizer.decode(int(ids[i+1])).replace(' ', '')
                kv_pairs.append((context, target))
            dstore_keys[offset:offset+actual_length] = \
                embed[:actual_length].cpu().numpy().astype(np.float16)
            # TODO: maybe values should be stored as int16?
            dstore_vals[offset:offset+actual_length] = \
                ids[1:1+actual_length].view(-1, 1).cpu().numpy().astype(np.int32)
            offset += actual_length
        if first: 
            print(kv_pairs[:10])

dstore_keys.flush()
dstore_vals.flush()

print('Finished saving vectors.')

with open('datastore/kv_pairs.p', 'wb+') as f:
    pickle.dump(kv_pairs, f)
