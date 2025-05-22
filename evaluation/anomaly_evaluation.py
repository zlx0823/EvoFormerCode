import pickle
import torch
from scipy import spatial
from scipy.stats import spearmanr
import datetime
from datetime import datetime as dt
import numpy as np
import pandas as pd

from graphert.temporal_embeddings import get_temporal_embeddings, get_embeddings_by_paths_average
from graphert.train_model import BertForMlmTemporalClassification, BertForTemporalClassification, \
    TemporalAttentionLayer, get_randompoc,get_graph_tokenizer

import warnings
warnings.filterwarnings("ignore")



def generate_poc_aixs(examlps,g_list):
    input_ids = examlps['input_ids']
    time = int(examlps['time'])
    pos_enc_list = []
    for word in input_ids:
        word = int(word)
        if word in [0,1,2,3]:
            pos_enc_list.append(torch.zeros(16))
        else:
            poc_emd = g_list[time][word]
            # print(poc_emd)
            pos_enc_list.append(poc_emd)
    pos_enc_list = torch.stack(pos_enc_list)
    # examlps['poc_emd'] = pos_enc_list
    return {'poc_emd': pos_enc_list}
def MRR(predicted: pd.DataFrame, real: pd.DataFrame):
    '''
    :param predicted: predicted similarity matrix by cosine similarity on the graphs embeddings
    :param real: true similarity matrix by MCS measure on the graphs pairs
    :return: average MRR across the time steps
    '''
    mrr = []
    for t, graph in predicted.iterrows():
        curr_real_t = real.loc[t]
        curr_real_t = curr_real_t.drop(t)
        top1_real = curr_real_t.sort_values(ascending=False).index[0]
        top1_predicted_rank = list(graph.sort_values(ascending=False).index).index(top1_real)
        if top1_predicted_rank == 0:
            top1_predicted_rank = 1
        mrr.append(1 / top1_predicted_rank)
    return np.mean(mrr)
def evaluate_anomalies(embs_vectors: np.array, times: list, anoms: list, google_df: pd.DataFrame = None):
    '''

    :param embs_vectors: temporal graph vectors for each time step. numpy array of shape (number of timesteps, graph vector dimension size)
    :param times: list of datetime of all graph's times
    :param anoms: list of anomalies times
    :param google: google trend data in case we have, with 'Day' and 'google' columns to represent the volume per day.
    :return:
    '''
    measures_df = pd.DataFrame(columns=['K', 'Recall', 'Precision'])
    ks = [5, 10]
    dist = np.array([spatial.distance.cosine(embs_vectors[i + 1], embs_vectors[i])
                     for i in range(1, len(embs_vectors) - 1)])

    for t,distace in zip(times,dist):
        print('{}:{}'.format(t,distace))
    for k in ks:
        top_k = (-dist).argsort()[:k]
        # 这里有问题
        top_k = np.array(times)[top_k]
        print('top_k:', top_k)
        tp = np.sum([1 if anom in top_k else 0 for anom in anoms])
        recall_val = tp / len(anoms)
        precision_val = tp / k
        measures_df = measures_df.append({'K': k, 'Recall': recall_val, 'Precision': precision_val},
                                         ignore_index=True)

    if google_df is not None:
        corr, pval = spearmanr(dist, google_df['google'].values[:-1])
        print(f'Spearman correlation: {corr}, p-value: {pval}')
    print(measures_df)




if __name__ == "__main__":
    print(torch.cuda.is_available())
    dataset_name = 'game_of_thrones'
    token_path = 'graph_tokenizer_poc_alltoken.tokenizer.json'
    model_path = f'../data/{dataset_name}/models/ep5_mask_walk32_model'

    # with open('../data/game_of_thrones/gameofthrones_2017_graphs_dynamic.pkl', 'rb') as f:
    #     graphs = pickle.load(f)

    with open('../data/formula/formula_2019_graphs_dynamic.pkl', 'rb') as f:
        graphs = pickle.load(f)
    graphs = {i: v for i, (k, v) in enumerate(graphs.items())}
    use_poc = False
    poc_list = []
    if use_poc:
        poc_list = get_randompoc(graphs, dataset_name, token_path)

    # with open('../data/game_of_thrones/gameofthrones_pocemd.pkl','wb') as f:
    #     pickle.dump(poc_list,f)
    # get temporal embeddings by the last layer
    # temporal_embeddings = get_temporal_embeddings(model_path)

    google_trends_df = pd.read_csv(f"../data/{dataset_name}/google_trends.csv",
                                   parse_dates=['Day'], date_parser=
                                   lambda x: dt.strptime(x, "%d-%m-%y"))

    # anoms = [datetime.date(2017, 7, 17), datetime.date(2017, 7, 24),
    #          datetime.date(2017, 7, 31),
    #          datetime.date(2017, 8, 7), datetime.date(2017, 8, 14), datetime.date(2017, 8, 21),
    #          datetime.date(2017, 8, 28)]

    # anoms = [datetime.date(2019, 3, 17), datetime.date(2019, 3, 31),
    #          datetime.date(2019, 4, 14),
    #          datetime.date(2019, 4, 28)]

    # # get temporal embeddings by averaging the paths embeddings per time
    random_walk_path = f'../data/{dataset_name}/paths_walk_len_32_num_walks_5.csv'
    data_df = pd.read_csv(random_walk_path, index_col=None)
    graph_tokenizer = get_graph_tokenizer(dataset_name, 32, token_path)
    data_df = data_df.sample(15_0000, random_state=42)
    data_df.drop('Unnamed: 0', axis=1, inplace=True)
    df_tokenized = data_df['sent'].apply(
        lambda examples: graph_tokenizer(examples, padding='max_length', truncation=True)).apply(pd.Series)
    data_df = pd.concat([data_df, df_tokenized], axis=1)
    data_df.drop(columns=["sent", "token_type_ids"], inplace=True)
    df_result = data_df.apply(lambda row: generate_poc_aixs(row, poc_list), axis=1).apply(pd.Series)
    data_df = pd.concat([data_df, df_result], axis=1)
    data_df['poc_emd'] = data_df['poc_emd'].apply(lambda x: x.tolist() if isinstance(x, torch.Tensor) else x)
    t_emb_mean, t_emb_weighted_mean, t_prob = get_embeddings_by_paths_average(data_df, model_path,
                                                                              dataset_name, walk_len=32,
                                                                              token_path=token_path, poc_list=poc_list,
                                                                              use_poc=True)
    print(t_prob)
    t_prob = [v for k, v in t_prob.items()]
    corr, pval = spearmanr(t_prob[1:], google_trends_df['google'].values)
    print(f'Spearman correlation: {corr}, p-value: {pval}')

