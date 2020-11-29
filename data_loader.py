import numpy as np
import networkx as nx
import random
import os
import torch

from torch.utils.data import Dataset


class DataSet(Dataset):

    def __init__(self, data_set_name):
        self.data_set_name = data_set_name
        self.path = './data/' + data_set_name + '/'
        self.node_attributes_exists = False
        self.graphs = self.build_graphs()

    def build_graphs(self):
        ds_a = np.genfromtxt(self.path + self.data_set_name + '_A.txt', delimiter=',', dtype=np.int)
        ds_graph_indicator = np.genfromtxt(self.path + self.data_set_name + '_graph_indicator.txt', dtype=np.int)
        ds_node_labels = np.genfromtxt(self.path + self.data_set_name + '_node_labels.txt', dtype=np.int)
        ds_graph_labels = np.genfromtxt(self.path + self.data_set_name + '_graph_labels.txt', dtype=np.int)
        if os.path.exists(self.path + self.data_set_name + '_node_attributes.txt'):
            self.node_attributes_exists = True
            ds_node_attributes = np.genfromtxt(self.path + self.data_set_name +
                                               '_node_attributes.txt', delimiter=',')

        graph_id, number_of_nodes = np.unique(ds_graph_indicator, return_counts=True)
        graphs = dict()

        for i, idx in enumerate(graph_id):
            graphs[idx - 1] = {'number_of_nodes': number_of_nodes[i]}

        stop = 0
        start = 1
        previous_number_of_lines = 0

        for key in graphs.keys():
            stop += graphs[key]['number_of_nodes']
            lines = np.array(np.where((start <= ds_a) & (ds_a <= stop)))
            number_of_lines = lines[0, -1] + 1 - previous_number_of_lines
            start_line = previous_number_of_lines
            previous_number_of_lines += number_of_lines
            stop_line = previous_number_of_lines - 1
            graph = nx.Graph()

            for i in range(start_line, stop_line + 1):
                edge = ds_a[i] - start
                graph.add_edge(edge[0], edge[1])

            for i in range(start-1, stop):
                node_label = ds_node_labels[i]
                graph.add_node(i - (start - 1), label=node_label)
                if self.node_attributes_exists:
                    graph.add_node(i - (start - 1), attribute=ds_node_attributes[i])

            start = stop + 1
            graphs[key]['nx_graph'] = graph
            graphs[key]['label'] = ds_graph_labels[key]

        return graphs

    def __len__(self):
        return len(list(self.graphs.keys()))

    def __getitem__(self, idx):
        adjacency_matrix = nx.adjacency_matrix(self.graphs[idx]['nx_graph']).todense()
        label = self.graphs[idx]['label'] - 1
        node_labels = np.array(list(nx.get_node_attributes(self.graphs[idx]['nx_graph'], 'label').values()))
        if self.node_attributes_exists:
            feature_matrix = np.array(list(nx.get_node_attributes(self.graphs[idx]['nx_graph'], 'attribute').values()))
            sample = {'adjacency_matrix': torch.tensor(adjacency_matrix.astype(np.float32), dtype=torch.float32),
                      'label': torch.tensor(label, dtype=torch.int),
                      'node_labels': torch.tensor(node_labels, dtype=torch.int),
                      'feature_matrix': torch.tensor(feature_matrix, dtype=torch.float32)
                      }
        else:
            sample = {'adjacency_matrix': torch.tensor(adjacency_matrix, dtype=torch.float32),
                      'label': torch.tensor(label, dtype=torch.int),
                      'node_labels': torch.tensor(node_labels, dtype=torch.int)
                      }

        return sample


class BuildDataSet(Dataset):
    def __init__(self, data_set_name):
        self.data_set_name = data_set_name
        self.path = './data/' + data_set_name + '/'
        self.node_attributes_exists = False
        self.graphs = self.build_graphs()
        self.graphs = list(self.graphs.values())
        random.shuffle(self.graphs)

    def build_graphs(self):
        ds_a = np.genfromtxt(self.path + self.data_set_name + '_A.txt', delimiter=',', dtype=np.int)
        ds_graph_indicator = np.genfromtxt(self.path + self.data_set_name + '_graph_indicator.txt', dtype=np.int)
        ds_node_labels = np.genfromtxt(self.path + self.data_set_name + '_node_labels.txt', dtype=np.int)
        number_of_node_labels = int(np.max(ds_node_labels)) - int(np.min(ds_node_labels)) + 1
        ds_graph_labels = np.genfromtxt(self.path + self.data_set_name + '_graph_labels.txt', dtype=np.int)

        if os.path.exists(self.path + self.data_set_name + '_node_attributes.txt'):
            self.node_attributes_exists = True
            ds_node_attributes = np.genfromtxt(self.path + self.data_set_name +
                                               '_node_attributes.txt', delimiter=',')

        graph_id, number_of_nodes = np.unique(ds_graph_indicator, return_counts=True)
        graphs = dict()

        for i, idx in enumerate(graph_id):
            graphs[idx - 1] = {'number_of_nodes': number_of_nodes[i]}

        stop = 0
        start = 1
        previous_number_of_lines = 0

        for key in graphs.keys():
            stop += graphs[key]['number_of_nodes']
            lines = np.array(np.where((start <= ds_a) & (ds_a <= stop)))
            number_of_lines = lines[0, -1] + 1 - previous_number_of_lines
            start_line = previous_number_of_lines
            previous_number_of_lines += number_of_lines
            stop_line = previous_number_of_lines - 1
            graph = nx.Graph()

            for i in range(start_line, stop_line + 1):
                edge = ds_a[i] - start
                graph.add_edge(edge[0], edge[1])

            for i in range(start-1, stop):
                node_label = ds_node_labels[i]
                graph.add_node(i - (start - 1), label=node_label)
                if self.node_attributes_exists:
                    graph.add_node(i - (start - 1), attribute=ds_node_attributes[i])
                else:
                    alternative_feature_matrix = self.one_hot_vector(number_of_node_labels, node_label)
                    graph.add_node(i - (start - 1), attribute=alternative_feature_matrix)

            start = stop + 1
            graphs[key]['nx_graph'] = graph
            graphs[key]['label'] = ds_graph_labels[key]

        return graphs

    @staticmethod
    def one_hot_vector(num_of_labels, label):
        vector = np.zeros((num_of_labels,))
        vector[label - 1] = 1.

        return vector

    def get_dataset(self):
        return self.graphs


class DataSetV2(Dataset):
    def __init__(self, data):
        self.graphs = data

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        adjacency_matrix = nx.adjacency_matrix(self.graphs[idx]['nx_graph']).todense()
        label = self.graphs[idx]['label'] - 1
        node_labels = np.array(list(nx.get_node_attributes(self.graphs[idx]['nx_graph'], 'label').values())) + 1
        feature_matrix = np.array(list(nx.get_node_attributes(self.graphs[idx]['nx_graph'], 'attribute').values()))

        if len(feature_matrix.shape) == 1:
            feature_label_matrix = np.concatenate((feature_matrix.reshape(-1, 1), node_labels.reshape(-1, 1)), axis=1)
        else:
            feature_label_matrix = np.concatenate((feature_matrix, node_labels), axis=1)

        feature_label_matrix = feature_label_matrix / feature_label_matrix.max(axis=0)

        sample = {'adjacency_matrix': torch.tensor(adjacency_matrix.astype(np.float32), dtype=torch.float32),
                  'label': torch.tensor(label, dtype=torch.int),
                  'node_labels': torch.tensor(node_labels, dtype=torch.int),
                  'feature_matrix': torch.tensor(feature_matrix, dtype=torch.float32),
                  'feature_label_matrix': torch.tensor(feature_label_matrix, dtype=torch.float32)
                  }
        return sample


class DataSetV3(Dataset):
    def __init__(self, data):
        self.graphs = data

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        if type(idx) == int:
            adjacency_matrix = nx.adjacency_matrix(self.graphs[idx]['nx_graph']).todense()
            label = int((self.graphs[idx]['label'] + 1) / 2)
            node_labels = np.array(list(nx.get_node_attributes(self.graphs[idx]['nx_graph'], 'label').values())) + 1
            feature_matrix = np.array(list(nx.get_node_attributes(self.graphs[idx]['nx_graph'], 'attribute').values()))

            sample = {'adjacency_matrix': torch.tensor(adjacency_matrix.astype(np.float32), dtype=torch.float32),
                      'label': torch.tensor(label, dtype=torch.int),
                      'node_labels': torch.tensor(node_labels, dtype=torch.int),
                      'feature_matrix': torch.tensor(feature_matrix, dtype=torch.float32)
                      }
            return sample
        elif type(idx) == slice:
            samples = []
            for i in range(idx.start, idx.stop):
                adjacency_matrix = nx.adjacency_matrix(self.graphs[i]['nx_graph']).todense()
                label = int((self.graphs[i]['label'] + 1) / 2)
                node_labels = np.array(list(nx.get_node_attributes(self.graphs[i]['nx_graph'], 'label').values()))
                feature_matrix = np.array(list(nx.get_node_attributes(self.graphs[i]['nx_graph'],
                                                                      'attribute').values()))

                sample = {'adjacency_matrix': torch.tensor(adjacency_matrix.astype(np.float32), dtype=torch.float32),
                          'label': torch.tensor(label, dtype=torch.int),
                          'node_labels': torch.tensor(node_labels, dtype=torch.int),
                          'feature_matrix': torch.tensor(feature_matrix, dtype=torch.float32)
                          }
                samples.append(sample)

            return samples

