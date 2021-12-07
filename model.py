# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, BertModel
from torch.nn import LayerNorm
# from knnlm import KNN_Dstore
from nn import MyTransformerDecoder, MyTransformerDecoderLayer, generate_square_subsequent_mask
import torch

class Model(nn.Module):
    def __init__(self, pretrained_weights, args):
        super(Model, self).__init__()
        self.args = args
        self.encoder = AutoModel.from_pretrained(pretrained_weights)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_weights)
        self.myembedding = MyEmbedding(self.encoder.embeddings)
        config = self.encoder.config
        if args.random_encoder:
            self.encoder = BertModel(config)
        elif args.no_encoder:
            self.encoder = None
        # print(config)
        decoder_layer = MyTransformerDecoderLayer(config.hidden_size, config.num_attention_heads,
                                                config.intermediate_size, dropout=0.1, activation='gelu')
        self.decoder = MyTransformerDecoder(decoder_layer, num_layers=args.decoder_layers, norm=LayerNorm(config.hidden_size))
        self.device = args.device
        self.copy_attention = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                            nn.Tanh() if not args.use_gelu else nn.GELU(),
                                            nn.Linear(config.hidden_size, config.hidden_size))
        self.linear_before_softmax = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                                   nn.Tanh() if not args.use_gelu else nn.GELU(),
                                                   nn.Linear(config.hidden_size, config.hidden_size))
        if args.pretrained_ar_path is not None:
            self.ar_model = ARModel(self.encoder.config.hidden_size * 2)
        else:
            self.ar_model = None

    def encode(self, source, no_context_update):
        encoder_output = self.encoder(
                     input_ids=source['input_ids'] if (not (self.args.dummy_source and no_context_update)) else source['input_ids'] * 0,
                     attention_mask=source['attention_mask'],
                     token_type_ids=source['token_type_ids'] if self.args.translate_backward is False else None)[0]
        if self.args.extra_encoder:
            encoder_output = self.extra_encoder(src=torch.transpose(encoder_output, 0, 1),
                                                mask=None,
                                                src_key_padding_mask=(source['attention_mask'] == 0))
            encoder_output = torch.transpose(encoder_output, 0, 1)
        return encoder_output

    def forward(self, data, target_input=None, no_encoder=None, 
                no_context_update=False, return_encoder_output=False, 
                encoder_output_saved=None, ret_last_ffn=False):
        source = {key: value.to(self.device) for key, value in data['source'].items()}
        target = {key: value.to(self.device) for key, value in data['target'].items()}
        label, choices = None, None
        if self.args.pointer_network:
            choices = {key: value.to(self.device) for key, value in data['choices'].items()}
            label = data['label'].to(self.device)
        if target_input is not None:
            target = target_input
        if encoder_output_saved is not None:
            encoder_output = encoder_output_saved
        elif not self.args.no_encoder:
            if no_context_update:
                with torch.no_grad():
                    encoder_output = self.encode(source, no_context_update)
            else:
                encoder_output = self.encode(source, no_context_update) # if not self.args.translate_backward else None)[0]
            # encoder_output *= 0

        if return_encoder_output:
            return encoder_output

        target_embedding = self.myembedding(target['input_ids'][:, :-1])
        target_length = target['input_ids'].shape[1]
        prediction = self.decoder(tgt=torch.transpose(target_embedding, 0, 1),
                                  memory=torch.transpose(encoder_output, 0, 1) if not self.args.no_encoder else None,
                                  tgt_mask=generate_square_subsequent_mask(target_length - 1).to(self.device),
                                  memory_mask=None,
                                  tgt_key_padding_mask=target['attention_mask'][:, :-1] == 0,
                                  memory_key_padding_mask=(source['attention_mask'] == 0) if not self.args.no_encoder else None,
                                  no_memory=self.args.no_encoder,
                                  no_context_update=False,
                                  ret_last_ffn=ret_last_ffn
                                  )
        if ret_last_ffn:
            prediction, last_ffn = prediction
            last_ffn = torch.transpose(last_ffn, 0, 1)
        prediction = torch.transpose(prediction, 0, 1)
        generation_prediction = self.linear_before_softmax(prediction)

        # pool info in encoder output and append to generation prediction
        # in this way, the context vector will have encoder info
        # knn_context will have shape [batch_size, seq_len, hid_size * 2]
        pooled_encoder_output = torch.sum(encoder_output * source['attention_mask'][..., None], dim=1)
        pooled_encoder_output /= torch.sum(source['attention_mask'], dim=1)[:, None]
        pooled_encoder_output = torch.cat([pooled_encoder_output[:, None]] * prediction.shape[1], dim=1)
        knn_context = torch.cat([pooled_encoder_output, generation_prediction], dim=-1)

        if self.args.pointer_network:
            choices_emb = self.myembedding.pembedding.word_embeddings(choices['input_ids'])
            logits = torch.einsum('bid, bjd->bij', prediction, choices_emb)
            logits = logits.masked_fill_(
                (choices['attention_mask'] == 0).unsqueeze(1).expand(-1, logits.shape[1], -1), float('-inf'))
        else:
            logits = torch.matmul(generation_prediction, torch.t(self.myembedding.pembedding.word_embeddings.weight))

        if not self.args.no_encoder and self.args.use_copy_attention:
            copy_prediction = self.copy_attention(prediction)
            copy_attention = torch.einsum('bid, bjd->bij', copy_prediction, encoder_output)
            if self.args.pointer_network:
                index = source['source_label']
            else:
                index = source['input_ids']
            copy_attention = copy_attention.masked_fill_(
                (source['attention_mask'] == 0).unsqueeze(1).expand(-1, copy_attention.shape[1], -1), 0)
            logits.scatter_add_(index=index.unsqueeze(1).expand(-1, logits.shape[1], -1),
                                src=copy_attention, dim=2)

        if ret_last_ffn:
            return logits, target, choices, label, prediction, last_ffn
        return logits, target, choices, label, prediction, knn_context


class KNNModel(Model):
    def __init__(self, dstore, *args, **kwargs):
        super(KNNModel, self).__init__(*args, **kwargs)
        self.dstore = dstore

    def get_knn_scores_per_step(self, x, save_knns=False):
        vocab_size = self.encoder.config.vocab_size
        pad_id = self.tokenizer.pad_token_id
        return self.dstore.get_knn_scores_per_step(
            x, vocab_size, pad_id, save_knns=save_knns
        )

    def interpolate(self, lprobs, knn_scores, last_ffn=None):
        # import pdb; pdb.set_trace()
        # taken from knnmt/fairseq/sequence_generator.py
        # lprobs = torch.stack([lprobs.squeeze(dim=1),
        #                       knn_scores.to(lprobs)], dim=0)
        last_lprobs = torch.stack([lprobs[:, -1], knn_scores.to(lprobs)], dim=0)
        coeffs = torch.ones_like(last_lprobs)
        lam = self.dstore.lmbda
        if self.ar_model is not None:
            lam = self.ar_model(last_ffn.view(-1, self.encoder.config.hid_size * 2))
            lam = lam.reshape(lprobs.shape)
        coeffs[0] = np.log(1 - lam)
        coeffs[1] = np.log(lam)
        last_lprobs = torch.logsumexp(last_lprobs + coeffs, dim=0)
        lprobs[:, -1] = last_lprobs
        # import pdb; pdb.set_trace()
        # lprobs is log of interpolated probability distribution
        return lprobs


class MyEmbedding(nn.Module):
    def __init__(self, embedding):
        super(MyEmbedding, self).__init__()
        self.pembedding = embedding

    def forward(self, input_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.pembedding.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.pembedding.word_embeddings(input_ids)
        position_embeddings = self.pembedding.position_embeddings(position_ids)

        embeddings = inputs_embeds + position_embeddings
        embeddings = self.pembedding.LayerNorm(embeddings)
        embeddings = self.pembedding.dropout(embeddings)
        return embeddings


class ARModel(nn.Module):
    def __init__(self, query_dim=1024, hidden_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, inputs):
        return F.log_softmax(self.fc(inputs), dim=1)
