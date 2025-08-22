import torch
import pandas as pd
from datasets import Dataset
import torch.nn as nn
import time
from torch.utils.data import DataLoader
from transformers import BertModel, AdamW
from transformers import BertForMaskedLM, DataCollatorForLanguageModeling, \
    TrainingArguments, Trainer, BertConfig, PreTrainedTokenizerFast, AutoModelForSequenceClassification
import math
import numpy as np
from datasets import load_metric
from tqdm import tqdm
from matplotlib import pyplot as plt
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertOnlyMLMHead
from torch.nn import CrossEntropyLoss

from JM_CODE.segmentationv3 import *
from evaluation.graph_similarity import *
from evaluation.similarity_ranknig_measures import *
# from evaluation.graph_similarity import graph2graph_mcs, graph2graph_similarity
# from evaluation.similarity_ranknig_measures import eval_similarity, precision_at_k, MAP_at_k, MRR, spearman_ranking, \
#     kendalltau_ranking
from graphert.transformer import Transormer_encoder
from graphert.processing_data import load_dataset
from graphert.train_tokenizer import train_graph_tokenizer
import scipy.sparse as sp
import networkx as nx
import pickle
import torch.nn.functional as F
# import plotly.graph_objects as go
from functools import partial
# from process_csv_data import *
from graphert.create_random_walks import create_random_walks
import os
import psutil
from itertools import islice
import shutil
import argparse
import warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
def create_mask_matrix(N, blocks):
    """
    创建掩码矩阵。
    参数:
    N: 矩阵大小 (NxN)
    blocks: 列表，每个元素是一个元组(start_row, size)，表示每个红色方块的起始行和大小
    返回:
    NxN 的掩码矩阵，红色方块表示为1，其他为0
    """
    # 创建一个全为0的数组，代表白色单元格
    mask_matrix = np.zeros((N, N), dtype=int)

    # 设置红色方块
    for start_row, size in blocks:
        mask_matrix[start_row:start_row + size, start_row:start_row + size] = 1
        x = mask_matrix[start_row:start_row + size, start_row:start_row + size]
        mask_matrix[start_row:start_row + size, start_row:start_row + size] = np.tril(x)

    return mask_matrix


def partition_intervals(lst):
    """
    将相同元素划分成不同的区间
    参数:
    lst: 输入列表
    返回:
    区间列表，每个区间为一个元组(start, end)
    """
    if not lst:
        return []

    intervals = []
    start = lst[0]

    for i in range(1, len(lst)):
        # 检查当前元素是否与前一个元素相同
        if lst[i] != lst[i - 1]:
            intervals.append((start, i - 1))
            start = i

    # 添加最后一个区间
    intervals.append((start, len(lst) - 1))

    return intervals

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Temporal_Graph_Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        # store encodings internally
        self.encodings = encodings

    def __len__(self):
        # return the number of samples
        return self.encodings['input_ids'].shape[0]

    def __getitem__(self, i):
        # return dictionary of input_ids, attention_mask, and labels for index i
        return {key: tensor[i] for key, tensor in self.encodings.items()}


class Layer(nn.Module):
    """Base layer class for PyTorch. Similar to the TensorFlow-based Layer class."""

    def __init__(self, name=None, logging=False):
        super(Layer, self).__init__()
        self.name = name if name else self.__class__.__name__.lower()
        self.logging = logging
        self.vars = nn.ParameterDict()  # Store learnable parameters

    def forward(self, inputs):
        """Wrapper for forward pass (computation graph)"""
        if self.logging:
            print(f'Logging inputs for layer {self.name}: {inputs}')
        outputs = self._call(inputs)
        if self.logging:
            print(f'Logging outputs for layer {self.name}: {outputs}')
        return outputs

    def _call(self, inputs):
        """To be implemented in subclasses."""
        raise NotImplementedError

    def _log_vars(self):
        """Logging model variables."""
        for name, param in self.vars.items():
            print(f'Variable {self.name}/{name}: {param}')


class TemporalAttentionLayer(Layer):
    def __init__(self, input_dim, n_heads, num_time_steps, attn_drop, residual=False, bias=True,
                 use_position_embedding=True, **kwargs):
        super(TemporalAttentionLayer, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.n_heads = n_heads
        self.num_time_steps = num_time_steps
        self.attn_drop = attn_drop
        self.residual = residual
        self.layer_norm1 = nn.LayerNorm(self.input_dim)
        self.layer_norm2 = nn.LayerNorm(self.input_dim)
        params = {"in_channels": self.input_dim, "out_channels": self.input_dim, "kernel_size": 1}
        self.lastcov = nn.Conv1d(**params)
        # Define position embeddings if used
        if use_position_embedding:
            self.vars['position_embeddings'] = nn.Parameter(
                torch.Tensor(num_time_steps, input_dim))
            nn.init.xavier_uniform_(self.vars['position_embeddings'])

        # Define Q, K, V weights for attention
        self.vars['Q_embedding_weights'] = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.vars['K_embedding_weights'] = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.vars['V_embedding_weights'] = nn.Parameter(torch.Tensor(input_dim, input_dim))

        nn.init.xavier_uniform_(self.vars['Q_embedding_weights'])
        nn.init.xavier_uniform_(self.vars['K_embedding_weights'])
        nn.init.xavier_uniform_(self.vars['V_embedding_weights'])

    def forward(self, inputs, seg_list):
        """Forward pass for temporal attention layer.
           inputs: Tensor of shape [N, T, F]
        """
        # Add position embeddings to input
        batch_size, T, C = inputs.size()
        position_inputs = torch.arange(self.num_time_steps, device=inputs.device).unsqueeze(0).repeat(batch_size, 1)
        temporal_inputs = inputs + F.embedding(position_inputs, self.vars['position_embeddings'])  # [N, T, F]

        # Query, Key, Value computation
        q = torch.matmul(temporal_inputs, self.vars['Q_embedding_weights'])  # [N, T, F]
        k = torch.matmul(temporal_inputs, self.vars['K_embedding_weights'])  # [N, T, F]
        v = torch.matmul(temporal_inputs, self.vars['V_embedding_weights'])  # [N, T, F]

        # Split into multiple heads and concatenate
        q_ = torch.cat(q.split(C // self.n_heads, dim=2), dim=0)  # [hN, T, F/h]
        k_ = torch.cat(k.split(C // self.n_heads, dim=2), dim=0)  # [hN, T, F/h]
        v_ = torch.cat(v.split(C // self.n_heads, dim=2), dim=0)  # [hN, T, F/h]

        # Scaled dot-product attention
        outputs = torch.matmul(q_, k_.transpose(-1, -2))  # [hN, T, T]
        outputs = outputs / (self.num_time_steps ** 0.5)

        # # Masked (causal) softmax to compute attention weights
        # mask = torch.tril(torch.ones(T, T, device=inputs.device)).unsqueeze(0).repeat(q_.size(0), 1, 1)  # [hN, T, T]
        # x = [[0, 0, 1, 1, 1, 2, 2, 2, 3, 3], [0, 0, 1, 1, 2, 2, 2, 3, 3, 3], [0, 0, 1, 1, 1, 2, 2, 2, 3, 3], [0, 0, 1, 1, 1, 2, 2, 2, 3, 3], [0, 0, 1, 1, 1, 2, 2, 3, 3, 3], [0, 0, 0, 1, 1, 1, 2, 2, 3, 3]]
        # x = [ v for lis in x for v in lis ]
        # x.append(len(x))
        N = len(seg_list)
        res = partition_intervals(seg_list)
        results = [(start, end - start + 1) for start, end in res]
        mask_matrix = create_mask_matrix(N, results)
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        mask = torch.from_numpy(mask_matrix).unsqueeze(0).repeat(q_.size(0), 1, 1).to(device)
        outputs = outputs.masked_fill(mask == 0, float('-inf'))
        outputs = F.softmax(outputs, dim=-1)  # Masked attention
        outputs = F.dropout(outputs, p=self.attn_drop, training=self.training)
        outputs = torch.matmul(outputs, v_)  # [hN, T, F/h]

        # Concatenate multiple heads
        split_outputs = torch.cat(torch.split(outputs, batch_size, dim=0), dim=-1)  # [N, T, F]
        split_outputs = self.feedforward(split_outputs)
        split_outputs = self.feedforward(split_outputs)
        # Optional: Feedforward and residual connection
        if self.residual:
            split_outputs += temporal_inputs
        split_outputs = self.layer_norm2(split_outputs)
        return split_outputs

    def feedforward(self, inputs):
        """Point-wise feedforward layer.
           inputs: A 3d tensor of shape [N, T, C]
        """
        outputs = F.relu(self.lastcov(inputs.transpose(1, 2))).transpose(1, 2)
        return outputs + inputs


class BertForTemporalClassification(BertPreTrainedModel):
    '''
    Train a model only for temporal classification with CrossEntropyLoss
    '''

    def __init__(self, config):
        super().__init__(config)
        self.temporal_num_labels = config.temporal_num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.temporal_num_labels)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            temporal_labels=None,
            output_attentions=None,
            output_hidden_states=None,
            poc_emd=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            poc_emd=poc_emd,
        )

        # temporal classification part
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss_fct = CrossEntropyLoss()
        temporal_loss = loss_fct(logits.view(-1, self.temporal_num_labels), temporal_labels.view(-1))

        loss = temporal_loss
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertForMlmTemporalClassification(BertPreTrainedModel):
    '''
        Train a model MLM for node masking and temporal classification.
        Use the temporal_weight to control the tradeoff between the two.

    '''

    def __init__(self, config, batch_size=32, walk_len=32, use_trsns=False, pred_label=None):
        super().__init__(config)
        self.relu = nn.ReLU()
        self.temporal_num_labels = config.temporal_num_labels
        self.vocab_size = config.vocab_size
        self.bert = BertModel(config, batch_size=batch_size, walk_len=walk_len)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.temporal_num_labels)
        self.pred = nn.Linear(2 * config.hidden_size, 2)
        # if use_trsns :
        #     for param in self.classifier.parameters():
        #         param.requires_grad = False
        self.mlm = BertOnlyMLMHead(config)
        self.temporal_weight = config.temporal_weight
        # self.fc = nn.Linear(32,384)
        # self.transformer = Transormer_encoder(1,self.temporal_num_labels,config.hidden_size,0.2,1)
        self.attentionlayer = TemporalAttentionLayer(input_dim=config.hidden_size, n_heads=4,
                                                     num_time_steps=self.temporal_num_labels, attn_drop=0.2)
        self.poc_fn = nn.Linear(16, config.hidden_size)
        self.init_weights()
        self.use_trsns = use_trsns
        self.pred_label = pred_label

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            temporal_labels=None,
            output_attentions=None,
            output_hidden_states=None,
            poc_emd=None,
            use_poc=False
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            poc_emd=poc_emd,
            use_poc=use_poc
        )

        # mlm part
        sequence_output = outputs[0]
        prediction_scores = self.mlm(sequence_output)
        loss_fct = CrossEntropyLoss()
        masked_lm_loss = loss_fct(prediction_scores.view(-1, self.vocab_size), labels.view(-1))

        # temporal classification part
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss_fct = CrossEntropyLoss()
        temporal_loss = loss_fct(logits.view(-1, self.temporal_num_labels), temporal_labels.view(-1))

        # 计算分类的准确率
        preds = torch.argmax(logits, dim=-1)
        correct = (preds == temporal_labels).float()

        accuracy = correct.mean()
        pred_loss = None
        classification = None
        if self.use_trsns:
            # # classfiction = self.classifier.weight.detach().clone()
            # # inputs = torch.unsqueeze(classfiction,0)

            # # inputs = torch.arange(self.temporal_num_labels).view(1, self.temporal_num_labels)

            # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # # inputs = inputs.to(device)
            # # output = self.transformer(inputs)

            # # output = output.view(-1,384)
            # classification = self.classifier.weight.detach().clone()
            # # classification[:60,:] = output
            # classification = classification.unsqueeze(0)
            # output = self.attentionlayer(classification)
            # # output = classification + output
            # output = output.squeeze(0)
            # self.classifier.weight = nn.Parameter(output,requires_grad = False)
            # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            classification = self.classifier.weight.detach().clone().to(device)
            x = self.classifier.weight.detach().clone().cpu().numpy()
            seg_list,_ = top_down(x, k=8)
            classification = classification.unsqueeze(0)
            classification = self.attentionlayer(classification, seg_list).squeeze(0)
            paired_embeddings = [torch.cat([classification[i], classification[i + 1]], dim=0) for i in
                                 range(classification.shape[0] - 1)]
            paired_embeddings = torch.stack(paired_embeddings)
            pred_logits = self.pred(paired_embeddings).to(device)
            pred_loss_fct = CrossEntropyLoss()
            self.pred_label = self.pred_label.to(device)
            # pred_logits = pred_logits.to(device)
            pred_loss = pred_loss_fct(pred_logits, self.pred_label)

        loss =    masked_lm_loss +  5  * temporal_loss

        if self.use_trsns:
            loss += 5 * pred_loss
        # loss =   masked_lm_loss +    temporal_loss

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        outputs = (loss, masked_lm_loss, temporal_loss, pred_loss) + outputs

        return outputs, accuracy, classification


def get_graph_tokenizer(dataset_name, walk_len, tokenizer_path):
    graph_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=f'../data/{dataset_name}/models/{tokenizer_path}', max_len=walk_len)
    graph_tokenizer.unk_token = "[UNK]"
    graph_tokenizer.sep_token = "[SEP]"
    graph_tokenizer.pad_token = "[PAD]"
    graph_tokenizer.cls_token = "[CLS]"
    graph_tokenizer.mask_token = "[MASK]"
    return graph_tokenizer


def train_mlm(dataset: Dataset, graph_tokenizer: PreTrainedTokenizerFast, dataset_name: str):
    '''
    Train only masking model for node-level making
    '''
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=graph_tokenizer, mlm=True, mlm_probability=0.15)

    training_args = TrainingArguments(
        output_dir="./",
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=32,
        save_steps=0,
        save_total_limit=0,
    )

    config = BertConfig(
        vocab_size=graph_tokenizer.vocab_size,
        hidden_size=384,
        num_hidden_layers=6,
        num_attention_heads=6,
        max_position_embeddings=64
    )

    model = BertForMaskedLM(config)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    trainer.train()

    trainer.save_model(f'../data/{dataset_name}/models/masking_model')


def tokenize_function(graph_tokenizer, examples, sent_col_name):
    return graph_tokenizer(examples[sent_col_name], padding='max_length', truncation=True)


def train_2_steps_model(random_walk_path, dataset_name, walk_len, sample_num=None):
    '''
    Train mlm and temporal model one after another, first train the mlm, then the classification. save torch model
    :param random_walk_path: file path to load the random walks corpus (created in create_random_walks.py)
    :param dataset_name:
    :param walk_len: length of a random walk, define the length of the sequence for the model
    :param sample_num: train using a sample number
    '''

    data_df = pd.read_csv(random_walk_path, index_col=None)
    graph_tokenizer = get_graph_tokenizer(dataset_name, walk_len)
    num_classes = len(set(data_df['time']))

    if sample_num:
        data_df = data_df.sample(sample_num)

    dataset = Dataset.from_pandas(data_df)
    dataset = dataset.map(lambda examples: tokenize_function(graph_tokenizer, examples, 'sent'), batched=True,
                          batch_size=512)

    cols = ['input_ids', 'token_type_ids', 'attention_mask']
    dataset.set_format(type='torch', columns=cols + ['time'])
    dataset = dataset.remove_columns(["sent", 'token_type_ids', 'Unnamed: 0'])

    train_mlm(dataset, graph_tokenizer, dataset_name)

    model = AutoModelForSequenceClassification.from_pretrained(f'datasets/{dataset_name}/models/masking_model/',
                                                               num_labels=num_classes)
    dataset = dataset.map(lambda examples: {'labels': list(examples['time'].numpy())}, batched=True)
    dataset.set_format(type='torch')

    dataset_test_dataset = dataset.train_test_split(test_size=0.2)
    train_dataset = dataset_test_dataset['train']
    test_dataset = dataset_test_dataset['test']
    metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(output_dir=f"datasets/{dataset_name}/output",
                                      per_device_train_batch_size=32,
                                      logging_strategy="steps",
                                      num_train_epochs=5,
                                      seed=0, save_strategy="epoch",
                                      )

    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_dataset, eval_dataset=test_dataset,
        compute_metrics=compute_metrics)

    trainer.train()
    torch.save(model, f'../data/{dataset_name}/models/time_classification_after_masking')


def generate_poc_aixs(examlps, g_list):
    input_ids = examlps['input_ids']
    time = examlps['time']
    pos_enc_list = []
    for t, sent in zip(time, input_ids):
        pos_enc = []
        for word in sent:
            word = int(word)
            if word in [0, 1, 2, 3]:
                pos_enc.append(torch.zeros(16))
            else:
                poc_emd = g_list[t][word]
                # print(poc_emd)
                pos_enc.append(poc_emd)
        pos_enc_list.append(torch.stack(pos_enc))
    # examlps['poc_emd'] = pos_enc_list
    return {'poc_emd': pos_enc_list}


def calculate_node_ratio(graphs):
    node_ratio = []
    total_nodes = sum([len(graph.nodes()) for graph in graphs])
    node_ratio = [len(graph.nodes()) / total_nodes for graph in graphs]
    return node_ratio


def train_mlm_temporal_model(random_walk_path: str, dataset_name: str, walk_len: int,
                             hidden_size: int, sample_num: int = None, g_list: list = None,
                             graphs=None, tokenizer_path: str = None, use_trsns: bool = False,
                             use_poc: bool = False, is_pretra: bool = False, model_path: str = None, epochs: int = 5,
                             pred_label=None, quiet=False):
    '''
    Train mlm and temporal model together (TM + MLM), save torch model
    :param random_walk_path: file path to load the random walks corpus (created in create_random_walks.py)
    :param dataset_name:
    :param walk_len: length of a random walk, define the length of the sequence for the model
    :param sample_num: train using a sample number
    '''
    df = pd.read_csv(random_walk_path, index_col=None)
    graph_tokenizer = get_graph_tokenizer(dataset_name, walk_len, tokenizer_path)
    # df['first'] = df['sent'].str.split().str[0]
    # df['count'] = df['sent'].str.split().apply(len)

    # 按比例采样
    # graph_list = [v for k,v in graphs.items()]
    # node_ratio = calculate_node_ratio(graph_list)

    # sample_nodes = [int(round(sample_num * node_ratio[i])) for i , value in enumerate(node_ratio)]
    # df_common = pd.DataFrame(columns = df.columns)

    # for t , v in enumerate(sample_nodes):
    #     df_time  = df[df['time'] == t].sample(v)
    #     df_common = pd.concat([df_common,df_time])

    # 找出公共节点
    # se_list = []
    # for i in range(-1,61,5):
    #     start = i+1
    #     end = i + 5 if i + 5 < 60 else 60
    #     se_list.append([start,end])

    # df_common = pd.DataFrame(columns = df.columns)
    # for start,end in se_list:
    #     gnode_list = [set(graphs.get(i).nodes()) for i in range(start,end+1)]
    #     time = set([i for i in range(start,end+1)])
    #     common_nodes = set(gnode_list[0].intersection(*gnode_list[1:]))
    #     df_windows = df[df['first'].isin(common_nodes) & df['time'].isin(time)]
    #     df_windows = df_windows.groupby(['time','first']).apply(lambda x : x.sample(5))
    #     df_windows = df_windows.reset_index(level = ['time','first'],drop = True)
    #     df_common = pd.concat([df_windows,df_common])

    # 找出零度节点
    # zero_degree = []
    # for time,graph in graphs.items():
    #     for node,degree in graph.degree():
    #         if degree == 0:
    #             zero_degree.append(node)
    # zero_degree = list(set(zero_degree))
    # df_zero = df[df['first'].isin(zero_degree)]
    # df_zero = df_zero[df_zero['count'] != 1]
    # df_zero = df_zero[(df_zero['p'] == 1) &(df_zero['q']==0.25)]
    # df_zero.drop_duplicates(subset = ['time','first'],keep = 'first',inplace = True)
    # print(len(df_zero))

    # df_no_degree = df[~df['first'].isin(zero_degree)]
    # print(len(df_no_degree))

    # filiter_df = df[df['count'] == 1]
    # filiter_df.drop_duplicates(subset=['time', 'first'], keep='first', inplace=True)
    # print(len(filiter_df))
    if sample_num:
        # df = pd.concat([df_zero,df_no_degree])
        # if sample_num > len(df_common):
        #     data_df = df.sample(sample_num - len(df_common))
        #     data_df = pd.concat([data_df,df_common])
        # else:
        #     data_df = df_common

        # data_df = data_df.sample(sample_num)
        # data_df.drop(['first', 'count'], axis=1, inplace=True)
        # data_df = df[(df['p'] == 1) & (df['q'] == 4)]
        data_df = df.sample(sample_num,random_state=42)
    dataset = Dataset.from_pandas(data_df)
    dataset = dataset.map(lambda examples: tokenize_function(graph_tokenizer, examples, 'sent'), batched=True,
                          batch_size=512)
    cols = ['input_ids', 'attention_mask']
    dataset = dataset.remove_columns(["sent", 'token_type_ids', 'Unnamed: 0'])
    dataset.set_format(type='torch', columns=cols + ['time', 'p', 'q'])
    if use_poc:
        new_process_values = partial(generate_poc_aixs, g_list=g_list)
        dataset = dataset.map(new_process_values, batched=True, batch_size=512)
        poc_emd = dataset['poc_emd']

    labels = dataset['input_ids']
    mask = dataset['attention_mask']
    temporal_labels = dataset['time']
    num_classes = len(set(dataset['time'].numpy()))
    # print(num_classes)
    # 掩码操作
    input_ids = labels.detach().clone()
    batch_ids = input_ids.tolist()
    sentences = graph_tokenizer.batch_decode(
        batch_ids,
        skip_special_tokens=False,  # 跳过 [CLS]/[SEP]/[PAD] 等特殊 token
        clean_up_tokenization_spaces=True  # 清理多余空格
    )
    sentences = [sent.split(' ') for sent in sentences]
    degree_lis = []
    for idx, sent in enumerate(sentences):
        degree = []
        for word in sent:
            if word not in ['[CLS]', '[SEP]', '[PAD]', '[UNK]']:
                degree.append(graphs[temporal_labels[idx].item()].degree(word) + 1)
            else:
                degree.append(0)
        degree_lis.append(degree)
    deg = torch.tensor(degree_lis, dtype=torch.float)
    # 每行独立归一化
    eps = 1e-8
    row_max = deg.max(dim=1, keepdim=True)[0]
    deg_norm = deg / (row_max + eps)

    base_mask_prob = 0.25
    mask_prob = base_mask_prob * deg_norm
    rand = torch.rand(input_ids.shape)
    mask_arr = rand < mask_prob
    selection = mask_arr.nonzero()
    total_masked = selection.shape[0]
    for sid in {
        graph_tokenizer.cls_token_id,
        graph_tokenizer.pad_token_id,
        graph_tokenizer.sep_token_id,
        graph_tokenizer.unk_token_id
    }:
        mask_arr &= (input_ids != sid)


    # 应用 mask
    input_ids[mask_arr] = graph_tokenizer.mask_token_id


    # mask_arr = (rand < .15) * (input_ids != graph_tokenizer.cls_token_id) * (
    #         input_ids != graph_tokenizer.pad_token_id) * (input_ids != graph_tokenizer.sep_token_id) * (
    #                    input_ids != graph_tokenizer.unk_token_id)
    # selection = ((mask_arr).nonzero())
    # input_ids[selection[:, 0], selection[:, 1]] = graph_tokenizer.mask_token_id

    if use_poc:
        d = Temporal_Graph_Dataset({'input_ids': input_ids, 'attention_mask': mask, 'labels': labels,
                                    'temporal_labels': temporal_labels, 'poc_emd': poc_emd
                                    })
    else:
        d = Temporal_Graph_Dataset({'input_ids': input_ids, 'attention_mask': mask, 'labels': labels,
                                    'temporal_labels': temporal_labels
                                    })
    batch_size = 32
    loader = torch.utils.data.DataLoader(d, batch_size=batch_size, shuffle=True)

    config = BertConfig(
        vocab_size=graph_tokenizer.vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=6,
        num_attention_heads=6,
        max_position_embeddings=walk_len + 4,
        temporal_num_labels=num_classes,
        temporal_weight=1
    )

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BertForMlmTemporalClassification(config, pred_label=pred_label, batch_size=batch_size, walk_len=walk_len,
                                             use_trsns=use_trsns).to(device)
    # model = torch.load('../data/facebook/models/poc_ep20_walk32_model').to(device)
    # for param in model.bert.parameters():
    #     param.requires_grad = False
    # for param in model.mlm.parameters():
    #     param.requires_grad = False
    model.train()
    optim = AdamW(model.parameters(), lr=1e-4)
    total_loss = []
    mlm_loss = []
    t_loss = []

    # 保存每个epoch后模型的评价指标
    best_Precision_5 = 0
    best_Precision_10 = 0
    best_map = 0
    best_mrr = 0
    sp = []
    p = []
    best_avg_score = 0
    for epoch in range(epochs):
        # if sample_num:
        #     data_df = df.sample(sample_num)
        # dataset = Dataset.from_pandas(data_df)
        # dataset = dataset.map(lambda examples: tokenize_function(graph_tokenizer, examples, 'sent'), batched=True,batch_size =512)
        # cols = ['input_ids', 'attention_mask']
        # dataset = dataset.remove_columns(["sent", 'token_type_ids', 'Unnamed: 0'])
        # dataset.set_format(type='torch', columns=cols + ['time', 'p', 'q'])
        # labels = dataset['input_ids']
        # mask = dataset['attention_mask']
        # temporal_labels = dataset['time']
        # num_classes = len(set(dataset['time'].numpy()))
        # input_ids = labels.detach().clone()
        # rand = torch.rand(input_ids.shape)
        # mask_arr = (rand < .15) * (input_ids != graph_tokenizer.cls_token_id) * (input_ids != graph_tokenizer.pad_token_id) * (input_ids != graph_tokenizer.sep_token_id) * (input_ids != graph_tokenizer.unk_token_id)
        # selection = ((mask_arr).nonzero())
        # input_ids[selection[:, 0], selection[:, 1]] = graph_tokenizer.mask_token_id
        # d = Temporal_Graph_Dataset({'input_ids': input_ids, 'attention_mask': mask, 'labels': labels,'temporal_labels': temporal_labels})
        # loader = torch.utils.data.DataLoader(d, batch_size=batch_size, shuffle=True)
        loop = tqdm(loader, leave=False,ncols=150)
        temp_emb = None
        for i, batch in enumerate(loop):
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            poc_emd = torch.stack(batch['poc_emd']).to(device) if use_poc else None
            # poc_emd = batch['poc_emd'].to(device) if use_poc else None
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            # print(labels)
            temporal_labels = batch['temporal_labels'].to(device)
            # 如果需要位置编码，在方法中加入参数poc_emd = poc_emd
            outputs, accuracy, temp_emb = model(input_ids, attention_mask=attention_mask, labels=labels,
                                                temporal_labels=temporal_labels, poc_emd=poc_emd, use_poc=use_poc)
            # extract loss
            loss = outputs[0]
            loss.backward()
            optim.step()
            loop.set_description(f'Epoch {epoch}')
            if use_trsns:
                loop.set_postfix(loss=loss.item(), mlm_loss=outputs[1].item(), t_loss=outputs[2].item(),
                                 pre_loss=outputs[3].item(), accuracy=f"{accuracy.item() * 100:.2f}%")
            else:
                loop.set_postfix(loss=loss.item(), mlm_loss=outputs[1].item(), t_loss=outputs[2].item(),
                                 accuracy=f"{accuracy.item() * 100:.2f}%")
                # loop.set_postfix(loss=loss.item(), mlm_loss=outputs[1].item(), t_loss=outputs[2].item(),pre_loss = outputs[3].item(),accuracy = f"{accuracy.item() * 100:.2f}%")
            total_loss.append(loss.item())
            mlm_loss.append(outputs[1].item())
            t_loss.append(outputs[2].item())
            # if i % 1000 == 0:
            #     print(f'loss={np.mean(total_loss)}, mlm_loss={np.mean(mlm_loss)}, t_loss={np.mean(t_loss)}')
            #     total_loss = []
            #     mlm_loss = []
            #     t_loss = []
        # 加入评价指标
        similarity_matrix_gt = graph2graph_mcs(graphs)
        temporal_embeddings = model.classifier.weight.cpu().detach().numpy()
        # if not use_trsns:
        #     temporal_embeddings  = model.classifier.weight.cpu().detach().numpy()
        # else:
        #     temporal_embeddings = temp_emb.cpu().detach().numpy() * 0.3 + model.classifier.weight.cpu().detach().numpy() * 0.7
        graphs_dict = {}
        times = list(graphs.keys())
        for time, emb in enumerate(temporal_embeddings):
            t = times[time]
            graphs_dict[t] = emb
        predicted = graph2graph_similarity(graphs_dict)
        precision_5 = precision_at_k(predicted, similarity_matrix_gt, 5)
        precision_10 = precision_at_k(predicted, similarity_matrix_gt, 10)
        map = MAP_at_k(predicted, similarity_matrix_gt, 10)
        mrr = MRR(predicted, similarity_matrix_gt)

        # sp.append(spearman_ranking(predicted, similarity_matrix_gt))
        # p.append(kendalltau_ranking(predicted, similarity_matrix_gt))
        # print(f'这是epoch:{epoch}:')
        # print(eval_similarity(graph_embs=temporal_embeddings, times=list(graphs.keys()),
        #                       similarity_matrix_gt=similarity_matrix_gt))
        poc = 'poc_'
        tf = 'tf_'
        f = ''


        # torch.save(model, f'../data/{dataset_name}/models/mlm_and_temporal_64_3_tf_model')
        # torch.save(model, f'../data/{dataset_name}/models/test_model')
        if (precision_5 + precision_10 + map + mrr) / 4 > best_avg_score:
            best_Precision_5 = precision_5
            best_Precision_10 = precision_10
            best_map = map
            best_mrr = mrr
            best_avg_score = (precision_5 + precision_10 + map + mrr) / 4
            torch.save(model,
                       f'../data/{dataset_name}/models/{poc if use_poc else f}{tf if use_trsns else f}ep{epoch + 1}_mask_walk{walk_len}_bestmodel')
            print(f'eopoch:{epoch + 1}, best_Precision_5:{best_Precision_5},best_Precision_10:{best_Precision_10},best_map:{best_map},best_mrr:{best_mrr}')
        # torch.save(model,
        #            f'../data/{dataset_name}/models/{poc if use_poc else f}{tf if use_trsns else f}ep{epochs}_walk{walk_len}_model')


def train_only_temporal_model(random_walk_path: str, dataset_name: str, walk_len: int, sample_num: int = None):
    '''
    Train only temporal part (TM), save torch model
    :param random_walk_path: file path to load the random walks corpus (created in create_random_walks.py)
    :param dataset_name:
    :param walk_len: length of a random walk, define the length of the sequence for the model
    :param sample_num: train using a sample number

    '''
    data_df = pd.read_csv(random_walk_path, index_col=None)
    graph_tokenizer = get_graph_tokenizer(dataset_name, walk_len)

    if sample_num:
        data_df = data_df.sample(sample_num)

    dataset = Dataset.from_pandas(data_df)
    dataset = dataset.map(lambda examples: tokenize_function(graph_tokenizer, examples, 'sent'), batched=True,
                          batch_size=512)
    cols = ['input_ids', 'attention_mask']
    dataset = dataset.remove_columns(["sent", 'token_type_ids', 'Unnamed: 0'])
    dataset.set_format(type='torch', columns=cols + ['time', 'p', 'q'])

    num_classes = len(set(dataset['time'].numpy()))
    temporal_labels = dataset['time']
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    config = BertConfig(
        vocab_size=graph_tokenizer.vocab_size,
        hidden_size=384,
        num_hidden_layers=6,
        num_attention_heads=6,
        max_position_embeddings=walk_len + 2,
        temporal_num_labels=num_classes,
    )

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BertForTemporalClassification(config).to(device)
    optim = AdamW(model.parameters(), lr=1e-4, weight_decay=0.0001)
    epochs = 5

    for epoch in range(epochs):
        loop = tqdm(loader, leave=True)
        for i, batch in enumerate(loop):
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            temporal_labels = batch['time'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, temporal_labels=temporal_labels)
            # extract loss
            loss = outputs[0]
            loss.backward()
            optim.step()
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item(), )
    torch.save(model, f'datasets/{dataset_name}/models/only_temporal')


def get_randompoc(graphs, dataset_name, tokenizer_path):
    poc_list = []
    for key, g in tqdm(graphs.items(), desc="Processing pembd"):
        # print(len(g.nodes()))
        A = nx.adjacency_matrix(g)
        degrees = np.array([val for (node, val) in g.degree()])
        Dinv = sp.diags(degrees.clip(1) ** -1.0, dtype=float)
        RW = A * Dinv
        M = RW
        nb_pos_enc = 16
        PE = [torch.from_numpy(np.diag(M.toarray())).float()]
        M_power = M
        for _ in range(nb_pos_enc - 1):
            M_power = M_power * M
            PE.append(torch.from_numpy(np.diag(M_power.toarray())).float())
        PE = torch.stack(PE, dim=-1)
        graph_tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=f'../data/{dataset_name}/models/{tokenizer_path}', max_len=64)
        graph_tokenizer.unk_token = "[UNK]"
        graph_tokenizer.sep_token = "[SEP]"
        graph_tokenizer.pad_token = "[PAD]"
        graph_tokenizer.cls_token = "[CLS]"
        graph_tokenizer.mask_token = "[MASK]"
        poc_dic = {graph_tokenizer.convert_tokens_to_ids(str(node)): PE[i] for i, node in enumerate(g.nodes())}
        poc_list.append(poc_dic)
    return poc_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process input arguments.')
    parser.add_argument('--dataset_name', type=str, default='formula', help='Name of the dataset')
    parser.add_argument('--sample_num', type=int, default=100000, help='Sample number')
    parser.add_argument('--use_trsns', type=str2bool, default=False, help='Use trsns (True or False)')
    parser.add_argument('--use_poc', type=str2bool, default=False, help='Use poc (True or False)')
    parser.add_argument('--is_pretra', type=str2bool, default=False, help='Is pretra (True or False)')
    parser.add_argument('--epochs', type=int, default=5, help='epoch number')

    args = parser.parse_args()
    dataset_name = args.dataset_name
    walk_len = 32
    print('This is the dataset: {}'.format(dataset_name))
    sample_num = args.sample_num
    use_trsns = args.use_trsns
    use_poc = args.use_poc
    is_pretra = args.is_pretra
    epochs = args.epochs
    model_path = '../data/formula/models/poc_ep5_walk32_model'
    random_walk_path = f'../data/{dataset_name}/paths_walk_len_{walk_len}_num_walks_{3 if walk_len == 64 else 5}.csv'

    tokenizer_path = 'graph_tokenizer_64_3.tokenizer.json' if walk_len == 64 else 'graph_tokenizer_poc_alltoken.tokenizer.json'
    # tokenizer_path = 'static_graph.json'
    # tokenizer_path = 'graph_tokenizer.tokenizer.json'
    if dataset_name == 'formula':
        with open('../data/formula/formula_2019_graphs_dynamic.pkl', 'rb') as f:
            graphs = pickle.load(f)

    if dataset_name == 'enron':
        graph_path = '../data/enron/out.enron'
        graph_df = pd.read_table(graph_path, sep=' ', header=None)
        graph_df.columns = ['source', 'target', 'weight', 'time']
        graph_nx, temporal_graph = load_dataset(graph_df, dataset_name, time_granularity='months')
        graphs = temporal_graph.get_temporal_graphs(min_degree=5)

    if dataset_name == 'game_of_thrones':
        with open('../data/game_of_thrones/gameofthrones_2017_graphs_dynamic.pkl', 'rb') as f:
            graphs = pickle.load(f)
        # with open('../data/game_of_thrones/game_static.pkl','rb') as f:
        #     graphs = pickle.load(f)

    if dataset_name == 'facebook':
        graph_path = '../data/facebook/facebook-wall.txt'
        graph_df = pd.read_table(graph_path, sep='\t', header=None)
        graph_df.columns = ['source', 'target', 'time']
        graph_nx, temporal_graph = load_dataset(graph_df, dataset_name, time_granularity='months')
        graphs = temporal_graph.get_temporal_graphs(min_degree=5)

    if dataset_name == 'slashdot':
        with open('../data/slashdot/slashdot_monthly_dynamic.pkl', 'rb') as f:
            graphs = pickle.load(f)

    if dataset_name == 'DBLP5':
        with open('../JM_CODE/data/DBLP/ego_list_5.pkl', 'rb') as f:
            graphs = pickle.load(f)
        graphs = {t : g for t,g in enumerate(graphs)}


    if dataset_name == 'Twitter_WorldCup':
        with open('../data/Twitter_WorldCup/dynamic_graph.pkl', 'rb') as f:
            graphs = pickle.load(f)
    if dataset_name == 'FB':
        with open('../JM_CODE/data/FB/ego_list_1.pkl', 'rb') as f:
            graphs = pickle.load(f)
        graphs = {t : g for t,g in enumerate(graphs)}
    # if dataset_name == 'ucl':
    #     graph_path = '../data/ucl/out.ucl'
    #     graph_df = pd.read_table(graph_path, sep=' ', header=None)
    #     graph_df.columns = ['source', 'target', 'weight', 'time']
    #     graph_nx, temporal_graph = load_dataset(graph_df, dataset_name, time_granularity='weeks')
    #     graphs = temporal_graph.get_temporal_graphs(min_degree=5)
    #
    # if dataset_name == 'SBM':
    #     graph_df = pd.read_csv('../data/SBM/sbm_50t_1000n_adj.csv')
    #     graph_nx, temporal_graph = SBMload_dataset(graph_df, dataset_name)
    #     graphs = temporal_graph.get_temporal_graphs(min_degree=5)
    #     graphs = {i: v for i, (k, v) in enumerate(graphs.items())}

    graphs = {i: v for i, (k, v) in enumerate(graphs.items())}

    # 计算预测标签
    pred_label = []
    p_graphs = [v for k, v in graphs.items()]
    for i in range(1, len(p_graphs)):
        prev_graph = p_graphs[i - 1]
        curr_graph = p_graphs[i]
        pre_edge_count = prev_graph.number_of_edges()
        curr_edge_count = curr_graph.number_of_edges()
        if curr_edge_count > pre_edge_count:
            pred_label.append(1)
        else:
            pred_label.append(0)
    pred_label = torch.tensor(pred_label)
    # train_graph_tokenizer(random_walk_path, dataset_name, walk_len,tokenizer_path)
    # folder_path = f'../data/{dataset_name}/graph_paths'
    #
    # # 删除文件夹中的所有内容
    # shutil.rmtree(folder_path)
    #
    # # 重新创建空文件夹
    # os.mkdir(folder_path)
    poc_list = []
    if use_poc:
        poc_list = get_randompoc(graphs, dataset_name, tokenizer_path)

    hidden_size = 384
    # train_only_temporal_model(random_walk_path, dataset_name, walk_len, sample_num=100_000)
    train_mlm_temporal_model(random_walk_path, dataset_name, walk_len,
                             hidden_size=hidden_size, sample_num=sample_num,
                             g_list=poc_list, graphs=graphs, tokenizer_path=tokenizer_path,
                             use_trsns=use_trsns, use_poc=use_poc, is_pretra=is_pretra, model_path=model_path,
                             pred_label=pred_label, epochs=epochs)

    # train_2_steps_model(random_walk_path, dataset_name, walk_len, sample_num=100_000)

    # #参数d敏感实验
    # hidden_size_list = [48,96,192,384]
    # p5 = []
    # mr = []
    # mp = []
    # similarity_matrix_gt = graph2graph_mcs(graphs)
    # times = list(graphs.keys())
    #
    # for hidden_size in hidden_size_list:
    #     print(hidden_size)
    #     train_mlm_temporal_model(random_walk_path, dataset_name, walk_len, hidden_size = hidden_size,sample_num=sample_num,g_list = poc_list,graphs = graphs,tokenizer_path = tokenizer_path,use_trsns = use_trsns,use_poc = use_poc)
    #     model_path = f'../data/{dataset_name}/models/test_model'
    #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #     model = torch.load(model_path, map_location=device)
    #     model.eval().to(device)
    #     temporal_embeddings = model.classifier.weight.cpu().detach().numpy()
    #     graphs_dict = {}
    #     for time, emb in enumerate(temporal_embeddings):
    #         t = times[time]
    #         graphs_dict[t] = emb
    #     predicted = graph2graph_similarity(graphs_dict)
    #     p5.append(precision_at_k(predicted, similarity_matrix_gt,5))
    #     mr.append(MRR(predicted, similarity_matrix_gt))
    #     mp.append(MAP_at_k(predicted, similarity_matrix_gt, 10))
    # print(p5)
    # print(mr)
    # print(mp)
    # plt.figure(figsize=(6,4))
    # plt.plot(hidden_size_list, p5, 'o-', color='orange', label='Precision_5')
    # plt.plot(hidden_size_list, mr, 'o-', color='green', label='MRR')
    # plt.plot(hidden_size_list, mp, 'o-',color= 'red',label = 'MAP')
    # plt.xticks(hidden_size_list)
    # plt.legend()

    # plt.title('Measure Values vs. d')
    # plt.xlabel('d')
    # plt.ylabel('Value')
    # plt.grid(True)
    # plt.savefig(f'../data/{dataset_name}/parameters_d.png')

    # # 参数walk_length敏感试验
    # qs = [0.25, 0.5, 1, 2, 4]
    # ps = [0.25, 0.5, 1, 2, 4]
    # p5 = []
    # mr = []
    # mp = []
    # similarity_matrix_gt = graph2graph_mcs(graphs)
    # times = list(graphs.keys())
    # num_walks_list = [5]
    # walk_length_list = [8,10,16,24,32]
    # random_walk_path = f'../data/{dataset_name}/parameters.csv'
    # for walk_length in walk_length_list:
    #     walk_lengths = [walk_length]
    #     wak_len = walk_length
    #     hidden_size = 384
    #     create_random_walks(graphs, ps, qs, walk_lengths, num_walks_list, dataset_name)
    #     train_mlm_temporal_model(random_walk_path, dataset_name, walk_len, hidden_size = hidden_size,sample_num=sample_num,g_list = poc_list,graphs = graphs,tokenizer_path = tokenizer_path,use_trsns = use_trsns,use_poc = use_poc)
    #     model_path = f'../data/{dataset_name}/models/test_model'
    #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #     model = torch.load(model_path, map_location=device)
    #     model.eval().to(device)
    #     temporal_embeddings = model.classifier.weight.cpu().detach().numpy()
    #     graphs_dict = {}
    #     for time, emb in enumerate(temporal_embeddings):
    #         t = times[time]
    #         graphs_dict[t] = emb
    #     predicted = graph2graph_similarity(graphs_dict)
    #     p5.append(precision_at_k(predicted, similarity_matrix_gt,5))
    #     mr.append(MRR(predicted, similarity_matrix_gt))
    #     mp.append(MAP_at_k(predicted, similarity_matrix_gt, 10))
    #     print(p5)
    #     print(mr)
    #     print(mp)
    # print(p5)
    # print(mr)
    # print(mp)
    # plt.figure(figsize=(6,4))
    # plt.plot(walk_length_list, p5, 'o-', color='orange', label='Precision_5')
    # plt.plot(walk_length_list, mr, 'o-', color='green', label='MRR')
    # plt.plot(walk_length_list, mp, 'o-',color= 'red',label = 'MAP')

    # plt.legend()

    # plt.title('Measure Values vs. walk_length')
    # plt.xlabel('walk_length')
    # plt.ylabel('Value')
    # plt.savefig(f'../data/{dataset_name}/parameters_walk_length.png')

    # #参数num_walks敏感试验
    # qs = [0.25, 0.5, 1, 2, 4]
    # ps = [0.25, 0.5, 1, 2, 4]
    # p5 = []
    # mr = []
    # mp = []
    # similarity_matrix_gt = graph2graph_mcs(graphs)
    # times = list(graphs.keys())
    # num_walks = [5,10,15,20]
    # walk_lengths = [32]
    # random_walk_path = f'../data/{dataset_name}/parameters.csv'
    # for num_walk in num_walks:
    #     num_walks_list = [num_walk]
    #     wak_len = 32
    #     hidden_size = 384
    #     create_random_walks(graphs, ps, qs, walk_lengths, num_walks_list, dataset_name)
    #     train_mlm_temporal_model(random_walk_path, dataset_name, walk_len, hidden_size = hidden_size,sample_num=sample_num,g_list = poc_list,graphs = graphs,tokenizer_path = tokenizer_path,use_trsns = use_trsns,use_poc = use_poc)
    #     model_path = f'../data/{dataset_name}/models/test_model'
    #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #     model = torch.load(model_path, map_location=device)
    #     model.eval().to(device)
    #     temporal_embeddings = model.classifier.weight.cpu().detach().numpy()
    #     graphs_dict = {}
    #     for time, emb in enumerate(temporal_embeddings):
    #         t = times[time]
    #         graphs_dict[t] = emb
    #     predicted = graph2graph_similarity(graphs_dict)
    #     p5.append(precision_at_k(predicted, similarity_matrix_gt,5))
    #     mr.append(MRR(predicted, similarity_matrix_gt))
    #     mp.append(MAP_at_k(predicted, similarity_matrix_gt, 10))
    #     print(p5)
    #     print(mr)
    #     print(mp)
    # print(p5)
    # print(mr)
    # print(mp)
    # plt.figure(figsize=(6,4))
    # plt.plot(num_walks, p5, 'o-', color='orange', label='Precision_5')
    # plt.plot(num_walks, mr, 'o-', color='green', label='MRR')
    # plt.plot(num_walks, mp, 'o-', color='red', label='MAP')
    # plt.xticks(num_walks)
    # plt.legend()
    # plt.title('Measure Values vs. num_walks')
    # plt.xlabel('num_walks')
    # plt.ylabel('Value')
    # plt.grid(True,which='both',axis='y',linestyle='--',color='gray')
    # plt.savefig(f'../data/{dataset_name}/parameters_num_walks.png')

    # 可扩展实验1

    # edge_list = [1e2,1e3,1e4,1e5,1e6]
    # time_list = []
    # for edge_number in edge_list:
    #     start_time = time.time()
    #     m = edge_number
    #     n = int(2*m / 10)
    #     seed = 20160
    #     G = nx.gnm_random_graph(n, m, seed=seed)
    #     edges = list(G.edges())
    #     np.random.seed(seed)
    #     np.random.shuffle(edges)
    #     snapshots = np.array_split(edges, 10)
    #     subgraphs = [nx.Graph() for _ in range(10)]
    #     for i, snapshot in enumerate(snapshots):
    #         subgraphs[i].add_edges_from(snapshot)
    #     graphs = {i : v for i ,v in enumerate(subgraphs)}
    #     qs = [0.25, 0.5, 1, 2, 4]
    #     ps = [0.25, 0.5, 1, 2, 4]
    #     walk_lengths = [32]
    #     num_walks_list = [5]
    #     create_random_walks(graphs, ps, qs, walk_lengths, num_walks_list, dataset_name)
    #     end_time = time.time()

    #     cost_time = math.log10(end_time - start_time)
    #     time_list.append(cost_time)
    #     print(time_list)
    # plt.figure(figsize=(6,4))
    # plt.plot(hidden_size_list, p5, 'o-', color='orange', label='Precision_5')
    # plt.xticks(hidden_size_list)
    # plt.legend()
    # plt.xlabel('log10 of edges')
    # plt.ylabel('log10 of running seconds')
    # plt.grid(True)
    # plt.savefig(f'../data/{dataset_name}/parameters_d.png')

    # 可扩展实验2
    # len_graph_list = [5,10,20,30,40,50]
    # count_nondes = []
    # time_list = []
    # graphs1 = graphs
    # for len_graph in len_graph_list:
    #     graphs = dict(islice(graphs1.items(),len_graph))
    #     graphs_list = [set(v.nodes()) for i,v in graphs.items()]
    #     count_nondes.append(len(set.union(*graphs_list)))
    #     start = time.time()
    #     qs = [0.25, 0.5, 1, 2, 4]
    #     ps = [0.25, 0.5, 1, 2, 4]
    #     walk_lengths = [32]
    #     num_walks_list = [5]
    #     create_random_walks(graphs, ps, qs, walk_lengths, num_walks_list, dataset_name)
    #     end = time.time()
    #     count_time = math.log10(end - start)
    #     time_list.append(count_time)
    #     print(time_list)
    # print(time_list)
    # print(count_nondes)
    # plt.figure(figsize = (6,4))
    # plt.plot(len_graph_list, time_list, 'o-', color='orange', label='run time')
    # plt.xticks(len_graph_list)
    # plt.legend(loc='upper left',bbox_to_anchor=(1, 1))
    # plt.ylabel('log10 of running seconds')
    # for i, label in enumerate(count_nondes):
    #   plt.text(len_graph_list[i], plt.ylim()[0] - (plt.ylim()[1] - plt.ylim()[0]) * 0.05, label, ha='center')
    # plt.grid(True,which='both',axis='y',linestyle='--',color = 'gray')
    # plt.savefig(f'../data/{dataset_name}/time.png')















