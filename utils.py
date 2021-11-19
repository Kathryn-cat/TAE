# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import torch.nn.functional as F
import torch
from torch.nn.utils.rnn import pad_sequence
from babel.numbers import parse_decimal, NumberFormatError
from dataset_preprocessing.wikisql.lib.query import Query
import re
import unicodedata

num_re = re.compile(r'[-+]?\d*\.\d+|\d+')


def make_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--prefix', type=str, default='default')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--encoder_lr', type=float, default=1e-5)
    parser.add_argument('--decoder_lr', type=float, default=7.5e-5)
    parser.add_argument('--lambd', type=float, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--decoder_layers', type=int, default=4)
    parser.add_argument('--dataset_name', type=str, default='wikisql')
    parser.add_argument('--save_dir', type=str, default='/home/sajad/pretrain_sp_decocder')
    parser.add_argument('--just_evaluate', action='store_true', default=False)
    parser.add_argument('--just_initialize', action='store_true', default=False)
    parser.add_argument('--auxilary_lm', action='store_true', default=False)
    parser.add_argument('--valid_batch_size', type=int, default=50)
    parser.add_argument('--test_batch_size', type=int, default=32)
    parser.add_argument('--beam_num', type=int, default=10)
    parser.add_argument('--beam_search_base', type=int, default=3)
    parser.add_argument('--beam_search_alpha', type=float, default=0.9)
    parser.add_argument('--extra_encoder_layers', type=int, default=1)
    parser.add_argument('--early_stopping_epochs', type=int, default=20)
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--percentage', type=int, default=10)
    parser.add_argument('--language_model', action='store_true', default=False)
    parser.add_argument('--use_authentic_data', action='store_true', default=False)
    parser.add_argument('--use_tagged_back', action='store_true', default=False)
    parser.add_argument('--extra_encoder', action='store_true', default=False)
    parser.add_argument('--small_dataset', action='store_true', default=False)
    parser.add_argument('--combined_eval', action='store_true', default=False)
    parser.add_argument('--use_real_source', action='store_true', default=False)
    parser.add_argument('--combined_training', action='store_true', default=False)
    parser.add_argument('--create_mapping', action='store_true', default=False)
    parser.add_argument('--pointer_network', action='store_true', default=False)
    parser.add_argument('--gating', action='store_true', default=False)
    parser.add_argument('--extra_linear', action='store_true', default=True)
    parser.add_argument('--extra_copy_attention_linear', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--monolingual_ratio', type=float, default=1.)
    parser.add_argument('--beta', type=float, default=1.)
    parser.add_argument('--temp', type=float, default=1.)
    parser.add_argument('--mono_min_prob', type=float, default=.1)
    parser.add_argument('--label_smoothing', type=float, default=.1)
    parser.add_argument('--translate_backward', action='store_true', default=False)
    parser.add_argument('--copy_bt', action='store_true', default=False)
    parser.add_argument('--add_noise', action='store_true', default=False)
    parser.add_argument('--use_back_translation', action='store_true', default=False)
    parser.add_argument('--generate_back_translation', action='store_true', default=False)
    parser.add_argument('--no_encoder_update_for_bt', action='store_true', default=False)
    parser.add_argument('--just_analysis', action='store_true', default=False)
    parser.add_argument('--early_stopping', action='store_true', default=False)
    parser.add_argument('--use_copy_attention', action='store_true', default=True)
    parser.add_argument('--dummy_source', action='store_true', default=False)
    parser.add_argument('--dummy_question', action='store_true', default=False)
    parser.add_argument('--python', action='store_true', default=False)
    parser.add_argument('--EMA', action='store_true', default=True)
    parser.add_argument('--random_encoder', action='store_true', default=False)
    parser.add_argument('--sql_augmentation', action='store_true', default=False)
    parser.add_argument('--sql_where_augmentation', action='store_true', default=False)
    parser.add_argument('--use_column_type', action='store_true', default=False)
    parser.add_argument('--use_codebert', action='store_true', default=False)
    parser.add_argument('--fixed_copy', action='store_true', default=False)
    parser.add_argument('--combine_copy_with_real', action='store_true', default=False)
    parser.add_argument('--no_schema', action='store_true', default=False)
    parser.add_argument('--no_linear_opt', action='store_true', default=False)
    parser.add_argument('--fix_linear_layer', action='store_true', default=False)
    parser.add_argument('--use_gelu', action='store_true', default=True)
    parser.add_argument('--ema_param', type=float, default=.999)
    parser.add_argument('--bleu_threshold', type=float, default=0)
    parser.add_argument('--lr_decay', type=float, default=1)
    parser.add_argument('--n', type=int, default=256)
    parser.add_argument('--warmup_steps', type=int, default=0)
    return parser


def get_args(parser):
    args = parser.parse_args()
    if args.dataset_name == 'django' or args.dataset_name == 'conala':
        args.python = True
    elif args.dataset_name == 'wikisql':
        if not args.translate_backward:
            args.pointer_network = True
        args.beam_num = 5
        args.test_batch_size = 100
        args.valid_batch_size = 100
        args.eval_interval = 5
        args.beam_search_base = 0
        args.beam_search_alpha = 1
        args.early_stopping_epochs = args.early_stopping_epochs//args.eval_interval+1
        if args.small_dataset is False:
            args.epochs = 10
        else:
            args.epochs = 100

    elif args.dataset_name =='magic':
        args.eval_interval = 5
    elif args.dataset_name in ['atis', 'geo', 'imdb', 'scholar', 'advising', 'academic']:
        args.beam_search_base = 0
        args.beam_search_alpha = 1
    else:
        raise Exception("Wrong Dataset Name!")
    args.no_encoder = False
    if args.language_model:
        args.no_encoder = True

    if args.generate_back_translation:
        args.translate_backward = True
    
    return args


def preprocess_batch(data):
    data_intents = [d['intent'] for d in data]
    data_snippets = [d['snippet'] for d in data]
    keys = ['input_ids', 'attention_mask', 'token_type_ids']
    source_dict = {key: pad_sequence([torch.tensor(d[key]) for d in data_intents], batch_first=True, padding_value=0)
                              for key in keys}
    target_dict = {key: pad_sequence([torch.tensor(d[key]) for d in data_snippets], batch_first=True, padding_value=0)
                                for key in keys}
    return {'source': source_dict, 'target': target_dict}


def strip_accents(text):
    return ''.join(char for char in
                   unicodedata.normalize('NFKD', text)
                   if unicodedata.category(char) != 'Mn')


def my_annotate(sentence):
    gloss = []
    tokens = []
    after = []
    punctuation = {'.', ',', "'", '"', '/', '\\', '&', '*', '(', ')', '%', '$', '€', '£', '￥', '￥', '’', '–', '·', '—',
                   '-', '#', '!', '?', '+', '^', '=', ':', ';', '{', '}', '[', ']', '_'}
    word = ''
    for ind in range(len(sentence)):
        s = sentence[ind]
        if s == ' ':
            if len(word) > 0:
                gloss.append(word)
                after.append(' ')
                tokens.append(strip_accents(word.lower()))
                word = ''
            else:
                continue
        elif s in punctuation:
            if len(word)>0:
                gloss.append(word)
                after.append('')
                tokens.append(strip_accents(word.lower()))
                word = ''
            tokens.append(s)
            gloss.append(s)
            if ind < (len(sentence)-1) and sentence[ind+1] == ' ':
                after.append(' ')
            else:
                after.append('')
        else:
            word += s
    if len(word)>0:
        gloss.append(word)
        after.append('')
        tokens.append(strip_accents(word.lower()))
    return {'gloss': gloss, 'words': tokens, 'after': after}


def linear_combination(x, y, epsilon):
    return epsilon*x + (1-epsilon)*y


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


def LabelSmoothingCrossEntropy(preds, target, args, choices_attention=None):
    log_preds = F.log_softmax(preds, dim=1)
    nll = F.nll_loss(log_preds, target, reduction='none')
    if args.pointer_network and choices_attention is not None:
        coeff = (1/choices_attention.sum(1).float()).unsqueeze(1)
        masked_log_preds = log_preds.masked_fill(torch.isinf(log_preds), value=0.0)
        loss = -masked_log_preds.sum(dim=1)
        return linear_combination(loss * coeff, nll, args.label_smoothing)
    elif not args.pointer_network:
        loss = -log_preds.sum(dim=1) / preds.size()[1]
        return linear_combination(loss, nll, args.label_smoothing)
    else:
        return nll


def compute_loss(args, data, model, target_input=None, no_context_update=False, encoder_output_saved=None, test=False):
    target_input_model = None
    if target_input is not None:
        target_input_model = target_input
    logits, target, choices, labels, hidden, generation_prediction = (
        model(data, target_input=target_input_model, 
              no_context_update=no_context_update,
              encoder_output_saved=encoder_output_saved)
    )
    
    if test:
        knn_scores = model.get_knn_scores_per_step(generation_prediction)
        logits = model.interpolate(logits, knn_scores)

    if args.pointer_network:
        labels = labels[:, 1:target['input_ids'].shape[1]].to(args.device)
    else:
        labels = target['input_ids'][:, 1:]
    if target_input is not None:
        loss = None
    else:
        loss = LabelSmoothingCrossEntropy(torch.transpose(logits, 1, 2), labels, args,
                         choices['attention_mask'] if args.pointer_network else None)
        loss = (loss*target['attention_mask'][:, 1:])
    return loss, logits, choices


def generate_model_name(args):
    model_first_token = args.dataset_name
    extention = '_LM' if args.language_model is True else ''
    if extention == '_LM':
        if args.python:
            model_first_token = 'python'
        elif args.dataset_name == 'magic':
            model_first_token = 'java'
        else:
            model_first_token = 'sql'

    model_name = '{}_{}_model{}{}_combined_training={}_seed={}{}{}{}{}.pth'.format(
        model_first_token,
        args.prefix,
        extention,
        str(args.percentage) if args.small_dataset is True else '',
        args.combined_training,
        args.seed,
        '_beta=' + str(args.beta) if args.combined_training else '',
        '_tmp=' + str(args.temp) if args.combined_training else '',
        '_trns_back=' + str(args.translate_backward),
        '_use_backtr=' + str(args.use_back_translation) +
        '_lmd=' + str(args.lambd) +
        '_cp_bt=' + str(args.copy_bt) +
        '_add_no=' + str(args.add_noise) +
        '_no_en_upd=' + str(args.no_encoder_update_for_bt) +
        '_ratio=' + str(args.monolingual_ratio) +
        '_ext_li=' + str(args.extra_linear) +
        '_ext_cp_li=' + str(args.extra_copy_attention_linear) +
        '_cp_att=' + str(args.use_copy_attention) +
        '_EMA=' + str(args.EMA)[0] +
        '_rnd_enc=' + str(args.random_encoder)[0] +
        '_de_lr=' + str(args.decoder_lr) +
        '_mmp=' + str(args.mono_min_prob) +
        '_saug=' + str(args.sql_augmentation)[0] +
        '_dums=' + str(args.dummy_source)[0] +
        '_dumQ=' + str(args.dummy_question)[0] +
        '_rsr=' + str(args.use_real_source)[0] +
        '_fc=' + str(args.fixed_copy)[0] +
        '_ccr=' + str(args.combine_copy_with_real)[0]
    )
    return model_name


def get_next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def find_sub_sequence(sequence, query_seq):
    for i in range(len(sequence)):
        if sequence[i: len(query_seq) + i] == query_seq:
            return i, len(query_seq) + i
    raise IndexError


def my_detokenize_code(code, dictionary):
    code = code.replace('.', ' . ').replace(',', ' , ').replace("'", " ' ")\
               .replace('!', ' ! ').replace('"', ' " ').split()
    literal = []
    intent = dictionary['words']
    i = 0
    while i < len(code):
        index_i = -1
        max_length = 1
        for j in range(len(intent)):
            if code[i] == intent[j]:
                length = 1
                while (i+length) < len(code) and (j+length) < len(intent) and code[i+length] == intent[j+length]:
                    length += 1
                if length > max_length:
                    max_length = length
                    index_i = j
        if index_i == -1:
            literal.append(code[i]+' ')
            i += 1
        else:
            i += max_length
            for j in range(max_length):
                literal.append(dictionary['gloss'][index_i+j]+dictionary['after'][index_i+j])
    return ''.join(literal)


def my_detokenize(tokens, token_dict, raise_error=False):
    literal = []
    try:
        start_idx, end_idx = find_sub_sequence(token_dict['words'], tokens)
        for idx in range(start_idx, end_idx):
            literal.extend([token_dict['gloss'][idx], token_dict['after'][idx]])

        val = ''.join(literal).strip()
    except IndexError:
        if raise_error:
            raise IndexError('cannot find the entry for [%s] in the token dict [%s]' % (' '.join(tokens),
                                                                                        ' '.join(token_dict['words'])))
        for token in tokens:
            match = False
            for word, gloss, after in zip(token_dict['words'], token_dict['gloss'], token_dict['after']):
                if token == word:
                    literal.extend([gloss, after])
                    match = True
                    break

            if not match and raise_error:
                raise IndexError('cannot find the entry for [%s] in the token dict [%s]' % (' '.join(tokens),
                                                                                            ' '.join(
                                                                                                token_dict['words'])))
            if not match:
                literal.extend(token)
        val = ''.join(literal).strip()
    return val


def detokenize_query(query, tokenized_question, table_header_type):
    detokenized_conds = []
    for i, (col, op, val) in enumerate(query.conditions):
        val_tokens = val.split(' ')
        detokenized_cond_val = my_detokenize(val_tokens, tokenized_question)

        if table_header_type[col] == 'real' and not isinstance(detokenized_cond_val, (int, float)):
            if ',' not in detokenized_cond_val:
                try:
                    detokenized_cond_val = float(parse_decimal(detokenized_cond_val))
                except NumberFormatError as e:
                    try:
                        detokenized_cond_val = float(num_re.findall(detokenized_cond_val)[0])
                    except: pass
        detokenized_conds.append((col, op, detokenized_cond_val))
    detokenized_query = Query(sel_index=query.sel_index, agg_index=query.agg_index, conditions=detokenized_conds)
    return detokenized_query
