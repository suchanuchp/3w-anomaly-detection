# preprocess data
import numpy as np
import re
import pandas as pd
from typing import List
from util.const import LABEL_COL
# from utils.cons


def get_most_common_features(target, all_features, max = 3, min = 3):
    res = []
    main_keys = target.split('_')

    for feature in all_features:
        if target == feature:
            continue

        f_keys = feature.split('_')
        common_key_num = len(list(set(f_keys) & set(main_keys)))

        if common_key_num >= min and common_key_num <= max:
            res.append(feature)

    return res

def build_net(target, all_features):
    # get edge_indexes, and index_feature_map
    main_keys = target.split('_')
    edge_indexes = [
        [],
        []
    ]
    index_feature_map = [target]

    # find closest features(nodes):
    parent_list = [target]
    graph_map = {}
    depth = 2
    
    for i in range(depth):        
        for feature in parent_list:
            children = get_most_common_features(feature, all_features)

            if feature not in graph_map:
                graph_map[feature] = []
            
            # exclude parent
            pure_children = []
            for child in children:
                if child not in graph_map:
                    pure_children.append(child)

            graph_map[feature] = pure_children

            if feature not in index_feature_map:
                index_feature_map.append(feature)
            p_index = index_feature_map.index(feature)
            for child in pure_children:
                if child not in index_feature_map:
                    index_feature_map.append(child)
                c_index = index_feature_map.index(child)

                edge_indexes[1].append(p_index)
                edge_indexes[0].append(c_index)

        parent_list = pure_children

    return edge_indexes, index_feature_map


def split_to_instances(df):
    instances: List[pd.DataFrame] = []

    if 'start' not in df.columns:
        return [df]

    start_idx = df[df['start'] == 1].index
    if len(start_idx) > 1:
        for i in range(len(start_idx)-1):
            s = start_idx[i]
            t = start_idx[i+1]
            instances.append(df.iloc[s:t])
        instances.append(df.iloc[start_idx[-1]:])
    else:
        instances.append(df)
    return instances

def construct_data(data, feature_map, labels=0):
    res = []

    for feature in feature_map:
        if feature in data.columns:
            res.append(data.loc[:, feature].values.tolist())
        else:
            print(feature, 'not exist in data')
    # append labels as last
    sample_n = len(res[0])

    if type(labels) == int:
        res.append([labels]*sample_n)
    elif len(labels) == sample_n:
        res.append(labels)

    return res


def construct_data_v2(data, feature_map, labels=0):
    # want a list of data: [, labels]
    features = []  # [[x1, x2, ...], [x1, x2, ...], ...]
    lbs = []  # [[y1, y2, ...], [y1, y2, ...], ...]

    df_instances = split_to_instances(data)
    print('len')
    print(len(df_instances))
    for df in df_instances:
        xs = [df.loc[:, feature].values.tolist() for feature in feature_map]

        sample_n = len(xs[0])
        if type(labels) == int:
            lb = [labels] * sample_n
        else:
            lb = df[LABEL_COL].values.tolist()

        features.append(xs)
        print(np.max(lb))
        lbs.append(lb)

    return features, lbs


def build_loc_net(struc, all_features, feature_map=[]):

    index_feature_map = feature_map
    edge_indexes = [
        [],
        []
    ]
    for node_name, node_list in struc.items():
        if node_name not in all_features:
            continue

        if node_name not in index_feature_map:
            index_feature_map.append(node_name)
        
        p_index = index_feature_map.index(node_name)
        for child in node_list:
            if child not in all_features:
                continue

            if child not in index_feature_map:
                print(f'error: {child} not in index_feature_map')
                # index_feature_map.append(child)

            c_index = index_feature_map.index(child)
            # edge_indexes[0].append(p_index)
            # edge_indexes[1].append(c_index)
            edge_indexes[0].append(c_index)
            edge_indexes[1].append(p_index)
        

    
    return edge_indexes