'''
Train a simple NN that outputs KNN retrieval probability on context input.
'''

import os
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import Adam

import numpy as np
from dataset_preprocessing.conala import Conala
from utils import make_parser, get_args, preprocess_batch
from knnlm import KNN_Dstore
from model import Model, KNNModel, ARModel

parser = make_parser()
parser.add_argument('--pretrained_path', type=str, help='Path to pretrained model.')
parser.add_argument('--data_type', type=str,
    choices=['train', 'mined', 'code_only'], default='train',
    help='Type of data used in the store (each is a superset of the previous)')
parser.add_argument('--reg_coeff', type=float, default=0.05)
parser.add_argument('--lr', type=float, default=1e-4)
args = get_args(parser)

print(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dstore = KNN_Dstore(args)
model = KNNModel(dstore, 'bert-base-uncased', args)
model.to(device)
model.device = device
model.load_state_dict(torch.load(args.pretrained_path))
model.eval()

def load_dataset(args, tokenizer):
    splits = ['train', 'dev', 'test']
    datasets = []
    for split in splits:
        dataset = Conala(args.dataset_name, split, tokenizer, args)
        datasets.append(dataset)
    return (*datasets,) if len(datasets) > 1 else dataset

train_dataset, valid_dataset, test_dataset = load_dataset(args, model.tokenizer)
print(f'Validation data has {len(valid_dataset)} samples.')

loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                    num_workers=0, pin_memory=True, collate_fn=preprocess_batch)

# initialize adaptive retrieval model

ar_model = ARModel(query_dim=model.encoder.config.hidden_size * 2).to(device)

optimizer = Adam(ar_model.parameters(), lr=args.lr)

def compute_acc(probs, label):
    pred = torch.argmax(probs, dim=-1)
    acc = torch.mean((pred == label).to(float))
    return acc.detach().cpu().numpy().round(4)

for epoch_ix in tqdm(range(args.epochs)):
    for i, data in enumerate(tqdm(loader)):
        mask = data['target']['attention_mask']
        lengths = (mask.sum(dim=1) - 1).detach().cpu().numpy()
        logits, _, _, _, _, knn_query = model(data, ret_last_ffn=True)
        label = data['target']['input_ids'][:, 1:].to(device)
        tae_lprobs = F.log_softmax(logits, dim=-1) # [bs, seq_len, num_tokens]

        # sample a token in every sample
        # we are doing this so knn lookup doesn't blow up gpu memory
        sampled_indices = np.random.randint(lengths)
        tae_lprobs = tae_lprobs[np.arange(len(label)), sampled_indices]
        label = label[np.arange(len(label)), sampled_indices]
        knn_query = knn_query[np.arange(len(label)), sampled_indices]

        # convert labels to one hot
        label = label.flatten()
        label_onehot = torch.zeros(len(label), tae_lprobs.shape[-1]).to(device)
        label_onehot = label_onehot.scatter_(1, label[:, None], 1)
        label_onehot = label_onehot.reshape(tae_lprobs.shape)

        # get knn probs
        knn_lprobs = model.get_knn_scores_per_step(knn_query[:, None])

        lam = ar_model(knn_query / torch.linalg.norm(knn_query, dim=1, keepdim=True))
        lprobs = torch.stack([knn_lprobs, tae_lprobs], dim=1)
        combined_probs = torch.exp(lprobs + lam[..., None]).sum(dim=1)
        if np.isnan(torch.mean(lam).detach().cpu().numpy()):
            import pdb; pdb.set_trace()

        loss = torch.log((combined_probs * label_onehot).sum(-1)).mean()
        loss += args.reg_coeff * torch.mean(lam ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # log per-token accuracy
    acc_tae_only = compute_acc(tae_lprobs, label)
    acc_knn_only = compute_acc(knn_lprobs, label)
    acc_combined = compute_acc(combined_probs, label)
    print(f'[Ep {epoch_ix}] - acc_tae {acc_tae_only}; acc_knn {acc_knn_only}; acc_combined {acc_combined}')
    print(f'         avg_lambda {(torch.mean(torch.exp(lam[:, 0]))).detach().cpu().numpy().round(3)}')

    # save model
    save_path = os.path.join(args.save_dir, 'ar.pth')
    print(f'Saving model to {save_path}')
    torch.save(ar_model.state_dict(), save_path)
