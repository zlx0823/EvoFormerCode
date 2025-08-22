import networkx as nx
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB, LinExpr, quicksum as qsum
import time


def Zou_funtion_for_summary(graph_embedding):
    summary_vector = np.mean(graph_embedding, axis=0)
    norms = np.linalg.norm(graph_embedding, axis=1) * np.linalg.norm(summary_vector)
    cosine_similarity = np.dot(graph_embedding, summary_vector) / norms
    similarity = np.sum(cosine_similarity)
    return similarity, summary_vector


def Memoization(sub_segment, all_dis_dict, index_sub_segment):
    if len(sub_segment) == 1:
        best_index = index_sub_segment[0]
        best_val = 0.5

    if len(sub_segment) == 2:
        best_index = index_sub_segment[0]
        best_val = 1

    if len(sub_segment) == 3:
        error_list = []
        options = [[(index_sub_segment[0], index_sub_segment[0]), (index_sub_segment[1], index_sub_segment[2])],
                   [(index_sub_segment[0], index_sub_segment[1]), (index_sub_segment[2], index_sub_segment[2])]]
        for i, o in enumerate(options):
            if i == 0:
                if (o[1][0], o[1][1]) not in all_dis_dict:
                    error, representative = Zou_funtion_for_summary(sub_segment[1:])
                    all_dis_dict[(o[1][0], o[1][1])] = [error, representative]
                error_list.append(all_dis_dict[(o[0][0], o[0][1])][0] + all_dis_dict[(o[1][0], o[1][1])][0])
            if i == 1:
                if (o[0][0], o[0][1]) not in all_dis_dict:
                    error, representative = Zou_funtion_for_summary(sub_segment[0:2])
                    all_dis_dict[(o[0][0], o[0][1])] = [error, representative]
                error_list.append(all_dis_dict[(o[0][0], o[0][1])][0] + all_dis_dict[(o[1][0], o[1][1])][0])
        best_index = options[np.argmax(error_list)][0][1]
        best_val = np.max(error_list)

    if len(sub_segment) > 3:
        start = index_sub_segment[0]
        end = index_sub_segment[-1]
        index_list = []
        error_list = []
        if (index_sub_segment[1], end) not in all_dis_dict:
            error, representative = Zou_funtion_for_summary(sub_segment[1:])
            all_dis_dict[(index_sub_segment[1], end)] = [error, representative]
        index_list.append([(start, start), (index_sub_segment[1], end)])
        error_list.append(0.5 + all_dis_dict[(index_sub_segment[1], end)][0])
        for j in index_sub_segment[1:-2]:
            anchor_index = index_sub_segment.index(j)
            if (start, j) not in all_dis_dict:
                error, representative = Zou_funtion_for_summary(sub_segment[0:anchor_index + 1])
                all_dis_dict[(start, j)] = [error, representative]
            if (j + 1, end) not in all_dis_dict:
                error, representative = Zou_funtion_for_summary(sub_segment[anchor_index + 1:])
                all_dis_dict[(j + 1, end)] = [error, representative]
            index_list.append([(start, j), (j + 1, end)])
            error_list.append(all_dis_dict[(start, j)][0] + all_dis_dict[(j + 1, end)][0])
        if (start, index_sub_segment[-2]) not in all_dis_dict:
            error, representative = Zou_funtion_for_summary(sub_segment[0:len(index_sub_segment) - 1])
            all_dis_dict[(start, index_sub_segment[-2])] = [error, representative]
        index_list.append([(start, index_sub_segment[-2]), (end, end)])
        error_list.append(all_dis_dict[(start, index_sub_segment[-2])][0] + 0.5)

        best_index = index_list[np.argmax(error_list)][0][1]
        best_val = np.max(error_list)
    return best_index, best_val


def top_down(sequence, k):
    all_error = 0
    all_dis_dict = {}
    index_segment = list(range(len(sequence)))
    for i in range(len(sequence)):
        all_dis_dict[(i, i)] = [0.5, sequence[i]]
    ob_value_list = []
    cut_point_list = []
    for run in range(k - 1):
        if run == 0:
            cut_point, ob_value = Memoization(sequence, all_dis_dict, index_segment)
            index_segment = [index_segment[:cut_point + 1], index_segment[cut_point + 1:]]
            ob_value_list.append(ob_value)
            cut_point_list.append(cut_point)
        else:
            temporary_ob_value_list = []
            temporary_cut_point = []
            cut_which_segment = []
            for i in index_segment:
                cut_point, ob_value = Memoization(sequence[i[0]:i[-1] + 1], all_dis_dict, i)
                temporary_ob_value_list.append(ob_value)
                temporary_cut_point.append(cut_point)
                cut_which_segment.append(i)
            ob_value = np.max(temporary_ob_value_list)
            cut_point = temporary_cut_point[np.argmax(temporary_ob_value_list)]
            index_cut_point = cut_which_segment[np.argmax(temporary_ob_value_list)].index(cut_point)
            index_segment.remove(cut_which_segment[np.argmax(temporary_ob_value_list)])
            index_segment.append(cut_which_segment[np.argmax(temporary_ob_value_list)][:index_cut_point + 1])
            index_segment.append(cut_which_segment[np.argmax(temporary_ob_value_list)][index_cut_point + 1:])
            index_segment = sorted(index_segment[:])
            ob_value_list.append(ob_value)
            cut_point_list.append(cut_point)

    flat_reaults = [0] * len(sequence)
    for a, i in enumerate(sorted(index_segment)):
        for j in i:
            flat_reaults[j] = a

    N = len(sequence)

    means_index = []
    for i, j in enumerate(sorted(cut_point_list)):
        if i == 0:
            means_index.append((0, j))
        else:
            means_index.append((sorted(cut_point_list)[i - 1] + 1, j))

    if N > 1:
        means_index.append((sorted(cut_point_list)[-1] + 1, N - 1))

    summary_graphs = {}

    for i in means_index:
        summary_graphs[i] = all_dis_dict[i][1]

    return flat_reaults, summary_graphs