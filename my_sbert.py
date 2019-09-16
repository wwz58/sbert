from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import csv
from sklearn.metrics.pairwise import paired_cosine_distances
import torch
import torch.nn.functional as F
import re
import os
import unicodedata
import numpy as np
from sklearn import preprocessing
from torch.utils.data import Dataset, DataLoader
from torch import nn
from pytorch_transformers import BertModel, BertTokenizer, AdamW, WarmupLinearSchedule

import time
from scipy.stats import pearsonr, spearmanr
import gzip


class BertEmbedder(nn.Module):
    def __init__(self, pre_train_path, cache_dir=None):
        super(BertEmbedder, self).__init__()
        self.bert = BertModel.from_pretrained(
            pre_train_path, cache_dir=cache_dir)
        self.embed_dim = self.bert.config.hidden_size

    def forward(self, input_ids, input_mask):
        output_tokens = self.bert(
            input_ids=input_ids, attention_mask=input_mask)[0]
        return output_tokens


def mean_pool(token_embeddings, input_mask):
    input_mask_expanded = input_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = input_mask_expanded.sum(1)
    sum_mask = torch.clamp(sum_mask, min=1e-9)
    out = (sum_embeddings / sum_mask)
    return out


class ClfLoss(nn.Module):
    def __init__(self, dim, num_class=3):
        super(ClfLoss, self).__init__()
        self.num_class = num_class
        self.fc = nn.Linear(dim, num_class)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, u, v, label):
        scores = self.fc(torch.cat([u, v, (u-v).abs()], 1))
        loss = self.loss_fn(scores, label)
        return loss


def evaluate(model, loader, distance_fn=lambda a, b: 1 - paired_cosine_distances(a, b), device='cpu'):
    # def ,
    with torch.no_grad():
        model = model.to(device)
        model.eval()
        a_embeds, b_embeds = [], []
        # scores = []
        labels = []
        for batch in loader:
            batch = [e.to(device) for e in batch]
            a_input_ids, a_input_mask, b_input_ids, b_input_mask, label = batch

            a_embed, b_embed = model(a_input_ids, a_input_mask), model(
                b_input_ids, b_input_mask)
            a_embeds += a_embed.tolist()
            b_embeds += b_embed.tolist()
            # score = distance_fn(a_embed.cpu().numpy(), b_embed.cpu().numpy())
            # scores += score.tolist()
            labels += label.tolist()
    model.train()
    scores = distance_fn(np.asarray(a_embeds), np.asarray(b_embeds))
    return spearmanr(labels, scores)[0]


class SentTransformer(nn.Module):
    def __init__(self, sent_embeder, pooling_fn):
        super(SentTransformer, self).__init__()
        self.sent_embeder = sent_embeder
        self.pooling_fn = pooling_fn

    def forward(self, input_ids, input_mask):
        x_embed = self.sent_embeder(input_ids, input_mask)
        x_embed = self.pooling_fn(x_embed, input_mask)
        return x_embed

    def fit(self, loader, val_loader, num_epoch, print_every, eval_every, save_path, loss_fn, evaluate_fn, device='cpu'):
        self.loss_fn = loss_fn
        self = self.to(device)
        if device == 'cuda':
            self = nn.DataParallel(self)
        self.train()

        best_measure = 0
        global_step = 0
        total_step = num_epoch*len(loader)
        tic = time.time()

        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=2e-5, eps=1e-6, correct_bias=False)
        scheduler = WarmupLinearSchedule(
            optimizer, warmup_steps=0.1*total_step, t_total=total_step)

        for batch in loader:
            global_step += 1

            batch = [e.to(device) for e in batch]
            a_input_ids, a_input_mask, b_input_ids, b_input_mask, label = batch

            a_embed, b_embed = self(a_input_ids, a_input_mask), self(
                b_input_ids, b_input_mask)
            loss = loss_fn(a_embed, b_embed, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if (global_step+1) % print_every == 0:
                last = int(time.time()-tic)
                h, m, s = last//3600, last//60, last % 60
                print(
                    f'step {global_step+1}/{total_step}\tloss {loss:.4f}\tuse {h}h {m}m {s}s')
            if (global_step+1) % eval_every == 0:
                measure = evaluate_fn(self, val_loader, device=device)
                print(f'val measure: {measure:.4f}')
                if measure > best_measure:
                    best_measure = measure
                    if best_measure > 0.75:
                        if not os.path.exists(save_path):
                            os.makedirs(save_path, exist_ok=True)
                        fn = os.path.join(save_path, f'{measure:.4f}')
                        torch.save(self.state_dict(), fn)


class NLIReader(nn.Module):
    def __init__(self, dir='',  fn='train.gz', tokenize=None, sam=True):

        self.tokenize = tokenize
        label_map = {'neutral': 0, 'contradiction': 1, 'entailment': 2}
        self.s1 = gzip.open(os.path.join(dir, 's1.'+fn),
                            mode="rt", encoding="utf-8").readlines()
        self.s2 = gzip.open(os.path.join(dir, 's2.'+fn),
                            mode="rt", encoding="utf-8").readlines()
        self.label = gzip.open(os.path.join(dir, 'labels.'+fn),
                               mode="rt", encoding="utf-8").readlines()
        self.label = [label_map[l.rstrip()] for l in self.label]
        self.s1 = [l.rstrip() for l in self.s1]
        self.s2 = [l.rstrip() for l in self.s2]
        if sam:
            self.s1 = self.s1[:1000]
            self.s2 = self.s2[:1000]
            self.label = self.label[:1000]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        a, a_mask = self.tokenize(self.s1[idx])
        b, b_mask = self.tokenize(self.s2[idx])
        return a, a_mask, b, b_mask, self.label[idx]


class STSbReader(Dataset):
    def __init__(self, fn, tokenize, norm_label=True, sam=True):
        self.tokenize = tokenize
        self.a = []
        self.b = []
        self.label = []
        data = csv.reader(open(fn, encoding="utf-8"),
                          delimiter='\t', quoting=csv.QUOTE_NONE)
        for i, row in enumerate(data):
            self.a.append(row[5])
            self.b.append(row[6])
            score = float(row[4])
            if norm_label:
                score = (score - 0)/5
            self.label.append(score)
            if sam:
                if i > 1000:
                    break

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        a, a_mask = self.tokenize(self.a[idx])
        b, b_mask = self.tokenize(self.b[idx])
        return a, a_mask, b, b_mask, self.label[idx]


class MyBertTokenizer(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, seq):
        input_ids = self.tokenizer.encode(seq)
        # seq = '[CLS] ' + seq + ' [SEP]'
        # input_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(seq))
#         print(seq, input_ids)
        input_ids = self.tokenizer.add_special_tokens_single_sentence(
            input_ids)
        mask_ids = [1] * len(input_ids)
#         type_ids = np.asarray([0] * len(input_ids))
        return input_ids, mask_ids


def smart_collate_fn(batch, maxlen=128):
    s1_input_ids, s1_mask_ids, s2_input_ids, s2_mask_ids, labels = [], [], [], [], []
    datatype = None

    s1_max_len = min(maxlen, max([len(data[0]) for data in batch]))
    s2_max_len = min(maxlen, max([len(data[2]) for data in batch]))
    for s1_input, s1_mask, s2_input, s2_mask, label in batch:
        if datatype is None:
            if isinstance(label, int):
                datatype = torch.long
            elif isinstance(label, float):
                datatype = torch.float
        s1_paddings = [0] * (s1_max_len - len(s1_input))
        s2_paddings = [0] * (s2_max_len - len(s2_input))
        s1_input_ids.append(s1_input[:s1_max_len]+s1_paddings)
        s1_mask_ids.append(s1_mask[:s1_max_len]+s1_paddings)
        s2_input_ids.append(s2_input[:s2_max_len]+s2_paddings)
        s2_mask_ids.append(s2_mask[:s2_max_len]+s2_paddings)
        labels.append(label)
    s1_input, s1_mask = (torch.LongTensor(s1_input_ids),
                         torch.LongTensor(s1_mask_ids))
    s2_input, s2_mask = (torch.LongTensor(s2_input_ids),
                         torch.LongTensor(s2_mask_ids))
    labels = torch.Tensor(labels).to(datatype)
    return s1_input, s1_mask, s2_input, s2_mask, labels


cache_dir = '/home/wwz/pretrain_bert/bert-base-uncased/'
sent_embeder = BertEmbedder(cache_dir)
pool_fn = mean_pool
sent_transformer = SentTransformer(sent_embeder, pool_fn)

tokenizer = BertTokenizer.from_pretrained(cache_dir, do_lower_case=True)
tokenize = MyBertTokenizer(tokenizer)

# nli_set = NLIReader('../data/AllNLI/', 'train.gz', tokenize, sam=False)
# print(nli_set[0])
# train_loader = DataLoader(nli_set, batch_size=16,
#                           shuffle=True, collate_fn=smart_collate_fn)

stsb_set = STSbReader('../data/stsbenchmark/sts-test.csv', tokenize, sam=False)
print(stsb_set[0])
val_loader = DataLoader(stsb_set, batch_size=16,
                        shuffle=False, collate_fn=smart_collate_fn)

# sent_transformer.fit(train_loader, val_loader,
#                      1, 1000, 1000, 'output/train_nli', ClfLoss(sent_embeder.embed_dim*3), evaluate, device='cuda:0')

missing_keys, unexpected_keys = sent_transformer.load_state_dict(torch.load(
    'output/train_nli/0.7969'), strict=False)
print('missing_keys :', missing_keys, '\n',
      'unexpected_keys :', unexpected_keys)
test_mesure = evaluate(sent_transformer, val_loader, distance_fn=lambda a,
                       b: 1 - paired_cosine_distances(a, b), device='cuda:0')
print('test_measure: ', test_mesure)
