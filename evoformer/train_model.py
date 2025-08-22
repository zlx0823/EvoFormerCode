import torch
import pandas as pd
from datasets import Dataset
from fix_transformers import  AdamW
from fix_transformers import BertConfig, PreTrainedTokenizerFast
import numpy as np
from tqdm import tqdm
from evaluation.graph_similarity import *
from evaluation.similarity_ranknig_measures import *
from evaluation.graph_similarity import graph2graph_mcs, graph2graph_similarity
from evaluation.similarity_ranknig_measures import precision_at_k, MAP_at_k, MRR
from evoformer.processing_data import load_dataset
from evoformer.train_tokenizer import train_graph_tokenizer
import scipy.sparse as sp
import networkx as nx
import pickle
from functools import partial
import os
import shutil
import argparse
import warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


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


def get_graph_tokenizer(dataset_name, walk_len, tokenizer_path):
    graph_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=f'../data/{dataset_name}/models/{tokenizer_path}', max_len=walk_len)
    graph_tokenizer.unk_token = "[UNK]"
    graph_tokenizer.sep_token = "[SEP]"
    graph_tokenizer.pad_token = "[PAD]"
    graph_tokenizer.cls_token = "[CLS]"
    graph_tokenizer.mask_token = "[MASK]"
    return graph_tokenizer



def tokenize_function(graph_tokenizer, examples, sent_col_name):
    return graph_tokenizer(examples[sent_col_name], padding='max_length', truncation=True)


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
                             use_poc: bool = False, epochs: int = 5,
                             pred_label=None):
    '''
    Train mlm and temporal model together (TM + MLM), save torch model
    :param random_walk_path: file path to load the random walks corpus (created in create_random_walks.py)
    :param dataset_name:
    :param walk_len: length of a random walk, define the length of the sequence for the model
    :param sample_num: train using a sample number
    '''
    df = pd.read_csv(random_walk_path, index_col=None)
    graph_tokenizer = get_graph_tokenizer(dataset_name, walk_len, tokenizer_path)
    if sample_num:
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

    input_ids = labels.detach().clone()
    rand = torch.rand(input_ids.shape)

    mask_arr = (rand < .15) * (input_ids != graph_tokenizer.cls_token_id) * (
            input_ids != graph_tokenizer.pad_token_id) * (input_ids != graph_tokenizer.sep_token_id) * (
                       input_ids != graph_tokenizer.unk_token_id)
    selection = ((mask_arr).nonzero())
    input_ids[selection[:, 0], selection[:, 1]] = graph_tokenizer.mask_token_id

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

    model = EvoFomer(config, pred_label=pred_label, batch_size=batch_size, walk_len=walk_len,
                                             use_trsns=use_trsns).to(device)
    model.train()
    optim = AdamW(model.parameters(), lr=1e-4)
    total_loss = []
    mlm_loss = []
    t_loss = []

    # Save model evaluation metrics after each epoch
    best_Precision_5 = 0
    best_Precision_10 = 0
    best_map = 0
    best_mrr = 0
    best_avg_score = 0
    for epoch in range(epochs):
        loop = tqdm(loader, leave=False,ncols=150)
        for i, batch in enumerate(loop):
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            poc_emd = torch.stack(batch['poc_emd']).to(device) if use_poc else None
            # poc_emd = batch['poc_emd'].to(device) if use_poc else None
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            # print(labels)
            temporal_labels = batch['temporal_labels'].to(device)
            # If positional encoding is needed, add parameter poc_emd = poc_emd in the method
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
        # Add evaluation metrics
        similarity_matrix_gt = graph2graph_mcs(graphs)
        temporal_embeddings = model.classifier.weight.cpu().detach().numpy()
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
        poc = 'poc_'
        tf = 'tf_'
        f = ''
        if (precision_5 + precision_10 + map + mrr) / 4 > best_avg_score:
            best_Precision_5 = precision_5
            best_Precision_10 = precision_10
            best_map = map
            best_mrr = mrr
            best_avg_score = (precision_5 + precision_10 + map + mrr) / 4
            torch.save(model,
                       f'../data/{dataset_name}/models/{poc if use_poc else f}{tf if use_trsns else f}ep{epoch + 1}_mask_walk{walk_len}_bestmodel')
            print(f'eopoch:{epoch + 1}, best_Precision_5:{best_Precision_5},best_Precision_10:{best_Precision_10},best_map:{best_map},best_mrr:{best_mrr}')



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
    parser.add_argument('--epochs', type=int, default=5, help='epoch number')
    parser.add_argument('--hidden_size', type=int, default=384, help='Hidden size of the model')
    parser.add_argument('--walk_len', type=int, default=32, help='Length of the random walk sequence')

    args = parser.parse_args()
    dataset_name = args.dataset_name
    walk_len = args.walk_len
    print('This is the dataset: {}'.format(dataset_name))
    sample_num = args.sample_num
    use_trsns = args.use_trsns
    use_poc = args.use_poc
    epochs = args.epochs
    model_path = '../data/formula/models/poc_ep5_walk32_model'
    random_walk_path = f'../data/{dataset_name}/paths_walk_len_{walk_len}_num_walks_{3 if walk_len == 64 else 5}.csv'

    tokenizer_path = 'graph_tokenizer_64_3.tokenizer.json' if walk_len == 64 else 'graph_tokenizer_poc_alltoken.tokenizer.json'
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

    if dataset_name == 'facebook':
        graph_path = '../data/facebook/facebook-wall.txt'
        graph_df = pd.read_table(graph_path, sep='\t', header=None)
        graph_df.columns = ['source', 'target', 'time']
        graph_nx, temporal_graph = load_dataset(graph_df, dataset_name, time_granularity='months')
        graphs = temporal_graph.get_temporal_graphs(min_degree=5)

    if dataset_name == 'DBLP5':
        with open('../JM_CODE/data/DBLP/ego_list_5.pkl', 'rb') as f:
            graphs = pickle.load(f)
        graphs = {t : g for t,g in enumerate(graphs)}

    graphs = {i: v for i, (k, v) in enumerate(graphs.items())}

    # Calculate prediction labels
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
    train_graph_tokenizer(random_walk_path, dataset_name, walk_len,tokenizer_path)
    folder_path = f'../data/{dataset_name}/graph_paths'
    
    # Delete all contents in the folder
    shutil.rmtree(folder_path)

    # Recreate empty folder
    os.mkdir(folder_path)
    poc_list = []
    if use_poc:
        poc_list = get_randompoc(graphs, dataset_name, tokenizer_path)

    hidden_size = args.hidden_size
    train_mlm_temporal_model(random_walk_path, dataset_name, walk_len,
                             hidden_size=hidden_size, sample_num=sample_num,
                             g_list=poc_list, graphs=graphs, tokenizer_path=tokenizer_path,
                             use_trsns=use_trsns, use_poc=use_poc,
                             pred_label=pred_label, epochs=epochs)
















