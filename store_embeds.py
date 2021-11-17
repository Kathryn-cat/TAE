'''
Stores each context (i.e. intent/previous tokens) as a fixed-size embedding

Run with:
python3 store_embeds.py \
    --dstore_mmap datastore/train_store \
    --dataset_name conala \
    --pretrained_path {path/to/pretrained/model}

Afterward, run build_dstore.py with appropriate args to generate datastore.
'''

import os
import pdb
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import numpy as np
from dataset_preprocessing.conala import Conala
from utils import make_parser, get_args, preprocess_batch
from model import Model

parser = make_parser()
parser.add_argument('--pretrained_path', type=str, help='Location of pretrained model')
parser.add_argument('--data_type', type=str, 
    choices=['train', 'mined', 'code_only'], default='train', 
    help='Type of data used in the store (each is a superset of the previous)')
args = get_args(parser)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Model('bert-base-uncased', args)
model.to(device)
model.device = device
model.load_state_dict(torch.load(args.pretrained_path))
model.eval()

train_dataset = Conala('conala', 'train', model.tokenizer, args, monolingual=False)
train_loader = DataLoader(train_dataset, batch_size=args.test_batch_size, shuffle=False,
                            num_workers=0, pin_memory=False, collate_fn=preprocess_batch)

dstore_size = 0
for i in range(len(train_dataset)):
    example = train_dataset[i]
    dstore_size += (len(example['snippet']['input_ids']) - 1)
print('Total # of target tokens:', dstore_size)

if not os.path.isdir('datastore'):
    os.mkdir('datastore')

dstore_keys = np.memmap(f'datastore_{args.data_type}_keys.npy', dtype=np.float16, mode='w+',
                        shape=(dstore_size, model.encoder.config.hidden_size))
dstore_vals = np.memmap(f'datastore{args.data_type}_values.npy', dtype=np.int, mode='w+',
                        shape=(dstore_size, 1))

with torch.no_grad():
    offset = 0
    for i, data in enumerate(tqdm(train_loader)):
        lengths = data['target']['attention_mask'].sum(dim=1)
        *_, prediction, generation_prediction = model(data)
        input_ids = data['target']['input_ids']
        for pred, length, ids in zip(generation_prediction, lengths, input_ids):
            actual_length = length-1
            dstore_keys[offset:offset+actual_length] = \
                pred[:actual_length].cpu().numpy().astype(np.float16)
            dstore_vals[offset:offset+actual_length] = \
                ids[1:1+actual_length].view(-1, 1).cpu().numpy().astype(np.int)
            offset += length
        pdb.set_trace()

dstore_keys.flush()
dstore_vals.flush()

print('Finished saving vectors.')
