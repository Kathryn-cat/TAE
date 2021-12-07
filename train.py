# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from knnlm import KNN_Dstore
# import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
import argparse
import torch
from model import Model, KNNModel
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import pickle
from booster.utils import EMA
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR
from evaluation.evaluation import evaluate
from evaluation.compute_eval_metrics import compute_metric
from utils import compute_loss, get_next_batch
import os
from evaluation.evaluation import generate_hypothesis
from utils import generate_model_name, make_parser, get_args
from torch.nn.utils.rnn import pad_sequence
from dataset_preprocessing.django import Django
from dataset_preprocessing.conala import Conala
from dataset_preprocessing.small_sql import SmallSQL
from dataset_preprocessing.wikisql.wikisql import Wikisql


dataset_classes = {'django': Django,
                   'conala': Conala,
                   'atis': SmallSQL,
                   'geography': SmallSQL,
                   'wikisql': Wikisql}


def load_dataset(args, tokenizer):
    splits = ['train', 'dev', 'test']
    datasets = []
    for split in splits:
        dataset = dataset_classes[args.dataset_name](args.dataset_name, split, tokenizer, args)
        datasets.append(dataset)
    return (*datasets,) if len(datasets) > 1 else dataset


def preprocess_batch(data):
    data_intents = [d['intent'] for d in data]
    data_snippets = [d['snippet'] for d in data]
    keys = ['input_ids', 'attention_mask', 'token_type_ids']
    source_dict = {key: pad_sequence([torch.tensor(d[key]) for d in data_intents], batch_first=True, padding_value=0)
                              for key in keys}
    target_dict = {key: pad_sequence([torch.tensor(d[key]) for d in data_snippets], batch_first=True, padding_value=0)
                                for key in keys}
    extra_info = {}
    if args.pointer_network:
        source_dict['source_label'] = pad_sequence([torch.tensor(d['source_label']) for d in data_intents],
                                                   batch_first=True, padding_value=0)
        data_choices = [d['choices'] for d in data]
        extra_info['choices'] = {key: pad_sequence([torch.tensor(d[key]) for d in data_choices], batch_first=True, padding_value=0)
                              for key in keys}
        extra_info['label'] = pad_sequence([torch.tensor(d['label']) for d in data], batch_first=True, padding_value=0)
    return {'source': source_dict, 'target': target_dict, **extra_info}


def print_dataset_length_info(train_dataset):
    length = []
    for i in range(len(train_dataset)):
        length.append(len(train_dataset[i]['intent']['input_ids']))
    print("min", min(length))
    print("max", max(length))
    length = np.array(length)
    print('std', np.std(length))
    print('mean', np.mean(length))


def train(args):
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    pretrained_weights = "bert-base-uncased"
    model = Model(pretrained_weights, args)
    params_except_encoder = []
    for name, p in model.named_parameters():
        if not name.startswith('encoder.'):
            params_except_encoder.append(p)

    decoder_optimizer = torch.optim.Adam(lr=args.decoder_lr, params=params_except_encoder)
    encoder_optimizer = torch.optim.Adam(lr=args.encoder_lr, params=model.encoder.parameters())
    encoder_scheduler = ExponentialLR(encoder_optimizer, gamma=1)
    decoder_scheduler = LambdaLR(decoder_optimizer, lr_lambda=lambda step: (step+1)/args.warmup_steps if step<args.warmup_steps else args.lr_decay**(step-args.warmup_steps))
    model.to(args.device)
    if args.EMA:
        ema_model = EMA(model, args.ema_param)
    train_dataset, valid_dataset, test_dataset = load_dataset(args, model.tokenizer)
    # print_dataset_length_info(train_dataset)
    print('KNN Hyperparams:', 'k =', args.k, 'probe =', args.probe, 'lambda =', args.lmbda, 'temp =', args.knn_temp)
    if args.small_dataset:
        train_dataset = train_dataset[:round(len(train_dataset)*args.percentage/100)]
    else:
        args.percentage = 100
    if args.copy_bt:
        args.batch_size = int(args.batch_size//(1+args.monolingual_ratio))
    # print("Effective batch size", args.batch_size)
    model_name = generate_model_name(args)
    # print("model name", model_name)
    writer = SummaryWriter(log_dir=args.save_dir+'/logs/conalaexp/')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=preprocess_batch)
    valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=preprocess_batch)
    if args.copy_bt:
        copy_dataset = dataset_classes[args.dataset_name](name=args.dataset_name, split='train', tokenizer=model.tokenizer, args=args, monolingual=True)
        copy_loader = DataLoader(copy_dataset, batch_size=int(args.batch_size * args.monolingual_ratio),
                                 shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=preprocess_batch)
        copy_iter = iter(copy_loader)
        print('copy dataset size', len(copy_dataset))

    print('train set size:', len(train_dataset))
    print('dev set size', len(valid_dataset))
    print('test set size', len(test_dataset))
    # print("example of parallel data")
    # print(model.tokenizer.decode(train_dataset[0]['intent']['input_ids']))
    # print(model.tokenizer.decode(train_dataset[0]['snippet']['input_ids']))
    # if args.copy_bt:
        # print(len(copy_dataset))
        # print("example of monolingual data")
        # print(model.tokenizer.decode(copy_dataset[0]['intent']['input_ids']))
        # print(model.tokenizer.decode(copy_dataset[0]['snippet']['input_ids']))

    resume_file = os.path.join(args.save_dir, 'resume.pth')
    if not args.just_evaluate:
        if os.path.exists(resume_file):
            print("resume is loaded")
            checkpoint = torch.load(resume_file)
            model.load_state_dict(checkpoint['model_to_evaluate'])
            ema_model = EMA(model, args.ema_param)
            model.load_state_dict(checkpoint['model_to_train'])
            if not args.no_encoder:
                encoder_optimizer.load_state_dict(checkpoint['enc_optimizer_state'])
            decoder_optimizer.load_state_dict(checkpoint['dec_optimizer_state'])
            best_criteria = checkpoint['best_criteria']
            begin_epoch = checkpoint['epoch']
            early_stopping = checkpoint['early_stopping']
        else:
            best_criteria = -float('inf')
            begin_epoch = 0
            early_stopping = 0

        for epoch in range(begin_epoch, args.epochs):
            averaged_loss = 0
            print('Epoch :', epoch + 1, "Early Stopping:", early_stopping,
                  "encoder lr: ", encoder_scheduler.get_lr() if not args.no_encoder else "no encoder",
                  "decoder_lr", decoder_scheduler.get_lr())
            model.train()
            for data in tqdm(train_loader):
                loss, logits, choices = compute_loss(args, data, model)
                loss = loss.sum(1)
                if args.copy_bt:
                    copy_data = None
                    if args.copy_bt:
                        copy_data, copy_iter = get_next_batch(iterator=copy_iter, loader=copy_loader)
                        copy_data['source'] = copy_data['target']
                    loss_bt, _, _ = compute_loss(args, copy_data, model, no_context_update=args.no_encoder_update_for_bt)
                    loss = torch.cat([loss, loss_bt.sum(1)], dim=0)
                    if args.copy_bt:
                        del copy_data
                loss = loss.mean()
                averaged_loss += loss.item()*len(data['source']['input_ids'])
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                loss.backward()
                encoder_optimizer.step()
                decoder_optimizer.step()
                if args.EMA:
                    ema_model.update()
                decoder_scheduler.step()
            encoder_scheduler.step()
            averaged_loss = averaged_loss / len(train_loader.dataset)
            writer.add_scalar('Loss/train', averaged_loss, epoch)
            with torch.no_grad():
                model_to_evaluate = ema_model.model if args.EMA else model
                validation_loss = evaluate(args, valid_loader, model_to_evaluate, split='dev')
                print('validation loss', validation_loss)
                if (epoch + 1) % args.eval_interval == 0:
                    greedy_hype = generate_hypothesis(args, valid_loader, model_to_evaluate, search='greedy')
                    metrics, sampled_texts = compute_metric(greedy_hype, args.dataset_name, split='dev', tokenizer=model.tokenizer, args=args, return_data=True)
                    val_intents = [model.tokenizer.decode(item['intent']['input_ids']) for item in valid_dataset[:100]]
                    sampled_texts = [[y] + x for x, y in zip(sampled_texts, val_intents)]
                    writer.add_scalar('Loss/dev', validation_loss, epoch)
                    print('exact match accuracy', metrics['exact_match'])
                    print('bleu:', metrics['bleu'])
                    criteria = metrics['exec_acc'] if args.dataset_name == 'wikisql' \
                        else metrics['bleu'] if (args.dataset_name == 'conala' or args.dataset_name == 'magic')\
                        else metrics['exact_match']
                    print("criteria", criteria)
                    writer.add_scalar('evaluation metric', criteria, epoch)
                    if args.early_stopping:
                        if best_criteria < criteria:
                            best_criteria = criteria
                            torch.save(model_to_evaluate.state_dict(), os.path.join(args.save_dir, model_name))
                            early_stopping = 0
                        else:
                            early_stopping += 1
                        if early_stopping >= args.early_stopping_epochs:
                            break
                    else:
                        torch.save(model_to_evaluate.state_dict(), os.path.join(args.save_dir, model_name))
                    print("resume.pth is saved")
                    torch.save({
                        'epoch': epoch+1,
                        'model_to_evaluate': model_to_evaluate.state_dict(),
                        'model_to_train': model.state_dict(),
                        'enc_optimizer_state': encoder_optimizer.state_dict() if not args.no_encoder else None,
                        'dec_optimizer_state': decoder_optimizer.state_dict(),
                        'best_criteria': best_criteria,
                        'early_stopping': early_stopping
                    }, resume_file)

                    log_dict = metrics.copy()
                    log_dict.update({
                        'epoch': epoch,
                        'train_loss': averaged_loss,
                        'val_loss': validation_loss,
                        'criteria': criteria,
                        'text': wandb.Table(data=sampled_texts, columns=['Intent', 'GT', 'Pred'])
                    })
                    wandb.log(log_dict, step=epoch)


    with torch.no_grad():
        if args.knn:
            dstore = KNN_Dstore(args)
            model = KNNModel(dstore, pretrained_weights, args)
            model.to(args.device)
        model.load_state_dict(torch.load(os.path.join(args.save_dir, 'conala_weights.pth')))
        model.eval()
        valid_loader = DataLoader(valid_dataset, batch_size=args.test_batch_size, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True, collate_fn=preprocess_batch)
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True, collate_fn=preprocess_batch)

        loader = {'train': train_loader, 'dev': valid_loader, 'test': test_loader}
        for split in ['test']: #,'dev']:
            for search in ['greedy', 'beam']:
                if args.knn:
                    file = os.path.join(args.save_dir, 'knn_hype_{}_{}_{}_{}_{}.pt'.format(split, search, args.lmbda, args.k, args.probe))
                else:
                    file = os.path.join(args.save_dir, 'hype_{}_{}.pt'.format(split, search))
                print(file)
                # if os.path.exists(file):
                    # generated_set = pickle.load(open(file, 'rb'))
                # else:
                generated_set = generate_hypothesis(args, loader[split], model, search=search)
                with open(file, 'wb') as f:
                    pickle.dump(generated_set, f)
                metrics = compute_metric(generated_set, args.dataset_name, split=split, tokenizer=model.tokenizer, args=args)
                print('{} {} accuracy'.format(split, search), metrics['exact_match'])
                if search == 'beam':
                    print('{} {} oracle accuracy'.format(split, search), metrics['exact_oracle_match'])
                print('{} {} bleu score'.format(split, search), metrics['bleu'])
                print("{} {} exececution accuracy".format(split, search), metrics['exec_acc'])

                with open('results.txt', 'a') as f:
                    contents = f"{args.lmbda},{args.k},{args.probe},{metrics['exact_match']},{metrics['exact_oracle_match']},{metrics['bleu']}\n"
                    f.write(contents)

    writer.close()


if __name__ == '__main__':
    parser = make_parser()
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    import resource

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    args = get_args(parser)

    # wandb.init(name="knn-code-gen",
    #            config=vars(args))

    # print(args)
    train(args)
