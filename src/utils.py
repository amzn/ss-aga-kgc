import numpy as np
import torch
import pandas as pd
from torch_geometric.data import Data
from os.path import join
import torch
from tqdm import tqdm
from src.graph_sampler import k_hop_subgraph
from torch_geometric.data import DataLoader as Loader
import os


def save_model(model, output_dir, filename, args):
    """
    Save the trained knowledge model under output_dir. Filename: 'language.h5'
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # save the weights for the whole model
    ckpt_path = os.path.join(output_dir, filename)
    torch.save({
        'state_dict': model.state_dict(),
        'args': args,
    }, ckpt_path)

def get_negative_samples_alignment(batch_size_each, num_entity,num_negative=None):
    '''
    Generate one negative sample
    :param batch_size_each:
    :param num_entity:
    :return:
    '''
    if num_negative == None:
        rand_negs = torch.randint(high=num_entity, size=(batch_size_each,))  # [b,n]
    else:
        rand_negs = torch.randint(high=num_entity, size=(batch_size_each,num_negative))  # [b,n]

    return rand_negs


def load_model(ckpt_path, model, device):
    if not os.path.exists(ckpt_path):
        raise Exception("Checkpoint " + ckpt_path + " does not exist.")
    # Load checkpoint.
    checkpt = torch.load(ckpt_path)
    ckpt_args = checkpt['args']
    state_dict = checkpt['state_dict']
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(state_dict)
    # 3. load the new state dict
    model.load_state_dict(state_dict)
    model.to(device)


def get_negative_samples_graph(batch_size_each, num_entity):
    '''
    Generate one negative samaple
    :param batch_size_each:
    :param num_entity:
    :return:
    '''
    rand_negs = torch.randint(high=num_entity, size=(batch_size_each,))  # [b,1]

    return rand_negs



def Ranking_all_batch(predicted_t, embedding_matrix, k = None):
    '''
    Compute k-nearest neighbors in a batch
    If k== None, return ranked all candidatees
    otherwise, return top_k candidates
    :param predicted_t:
    :param embedding_matrix:
    :param k:
    :return:
    '''

    total_entity = embedding_matrix.shape[0]
    predicted_t = torch.unsqueeze(predicted_t, dim=1)  # [b,1,d]
    predicted_t = predicted_t.repeat(1, total_entity, 1)  # [b,n,d]

    distance = torch.norm(predicted_t - embedding_matrix, dim=2)  # [b,n]

    if k==None:
        k = total_entity

    top_k_scores, top_k_t = torch.topk(-distance, k=k)
    return top_k_t, top_k_scores




def get_language_list(data_dir):
    entity_dir = data_dir + "/entity"
    entity_files = list(os.listdir(entity_dir))
    entity_files = list(filter(lambda x: x[-3:] == "tsv", entity_files))
    entity_files = sorted(entity_files)
    print("Number of KGs is %d" % len(entity_files))

    kg_names = []
    for each_entity_file in entity_files:
        kg_name_each = each_entity_file[:2]
        kg_names.append(kg_name_each)

    return kg_names


def nodes_to_graph(sub_graph_list, node_index, batch_size=-1):
    one_batch = False
    if batch_size == -1:
        # get embeddings together without batch
        batch_size = node_index.shape[0]
        one_batch = True

    graphs = [sub_graph_list[i.item()] for i in node_index]

    graph_loader = Loader(graphs, batch_size=batch_size, shuffle=False)

    if one_batch:
        for one_batch in graph_loader:
            assert one_batch.edge_index.shape[1] == one_batch.edge_attr.shape[0]
            return one_batch
    else:
        return graph_loader


def nodes_to_graph_align(sub_graph_list, node_index, batch_size=-1):
    # node_index : [b,n_neg]
    one_batch = False
    if batch_size == -1:
        # get embeddings together without batch
        batch_size = node_index.shape[0]
        one_batch = True

    # Reshape into [b*n_neg,1]
    node_index = torch.reshape(node_index,(-1,))

    graphs = [sub_graph_list[i.item()] for i in node_index]

    graph_loader = Loader(graphs, batch_size=batch_size, shuffle=False)

    if one_batch:
        for one_batch in graph_loader:
            assert one_batch.edge_index.shape[1] == one_batch.edge_attr.shape[0]
            return one_batch
    else:
        return graph_loader



def create_subgraph_list(language, edge_index, edge_value, total_num_nodes, num_hops, k, node_base, relation_base):
    # Adding self-loop for nodes without edges
    # TODO: here k is for restricting the total number of edges in a subgrap. Wether to remove?
    # TODO padding self=edges for those do not have edges
    sub_graph_list = []
    num_edges = []

    node_list = [i for i in range(total_num_nodes)]

    for i in tqdm(node_list):

        [node_ids, edge_index_each, node_position, edge_masks] = k_hop_subgraph([i], num_hops, edge_index, num_nodes=total_num_nodes,
                                                  relabel_nodes=True)
        x = node_ids + node_base  # global indexing
        edge_index_each = edge_index_each[:, :k]
        # edge value can be zero!!!!!!!!!!!!!
        edge_value_masked = (edge_value + 1) * edge_masks
        edge_attr = edge_value_masked[edge_value_masked.nonzero(as_tuple=True)] - 1
        edge_attr = edge_attr[:k] + relation_base # global indexing

        assert edge_attr.shape[0] == edge_index_each.shape[1]

        node_position = torch.LongTensor([node_position])
        num_size = torch.LongTensor([len(node_ids)])
        graph_each = Data(x=x, edge_index=edge_index_each, edge_attr=edge_attr, y=node_position, num_size=num_size)
        sub_graph_list.append(graph_each)
        num_edges.append(edge_index_each.shape[1])

    print(language + ":Average subgraph edges %.2f" % np.mean(num_edges))


    return sub_graph_list




def get_kg_edges_for_each(data_dir, language, is_target_KG=False):
    '''
    TODO: whether include directional edges (1. do not incorporate. 2. adding the numbe of relation embeddings)
    :param data_dir:
    :param language:
    :param is_target_KG:
    :return:
    '''
    train_df = pd.read_csv(join(data_dir, language + '-train.tsv'), sep='\t', header=None,
                           names=['v1', 'relation', 'v2'])

    val_df = pd.read_csv(join(data_dir, language + '-val.tsv'), sep='\t', header=None,
                         names=['v1', 'relation', 'v2'])

    # Training data graph construction
    sender_node_list = train_df['v1'].values.astype(np.int).tolist()
    sender_node_list += train_df['v2'].values.astype(np.int).tolist()

    receiver_node_list = train_df['v2'].values.astype(np.int).tolist()
    receiver_node_list += train_df['v1'].values.astype(np.int).tolist()

    edge_weight_list = train_df['relation'].values.astype(np.int).tolist() + train_df['relation'].values.astype(
        np.int).tolist()

    # unified: Adding validation edges from supporter KG as well
    if not is_target_KG:
        sender_node_list += val_df['v1'].values.astype(np.int).tolist()
        sender_node_list += val_df['v2'].values.astype(np.int).tolist()

        receiver_node_list += val_df['v2'].values.astype(np.int).tolist()
        receiver_node_list += val_df['v1'].values.astype(np.int).tolist()

        edge_weight_list += val_df['relation'].values.astype(np.int).tolist()
        edge_weight_list += val_df['relation'].values.astype(np.int).tolist()

    edge_index = torch.LongTensor(np.vstack((sender_node_list, receiver_node_list)))
    edge_weight = torch.LongTensor(np.asarray(edge_weight_list))
    return edge_index, edge_weight


def get_subgraph_list(data_dir, language, is_target_KG, num_entity, num_hop, k, node_base, relation_base):

    edge_index, edge_type = get_kg_edges_for_each(data_dir + "/kg", language, is_target_KG=is_target_KG)

    sub_graph_list = create_subgraph_list(language, edge_index, edge_type, num_entity, num_hop, k, node_base, relation_base)

    return sub_graph_list



def subgrarph_list_from_alignment(seed_pairs,kg0,kg1,is_kg_list = False):
    '''
    lang0, lang1 alignment pairs. np.int

    Append nodes from other kgs to x
    :param seed_pairs:
    :param lang0:
    :param lang1:
    :return:
    '''
    # TOod: check them!

    num_relation = kg0.num_relation  # Total number of relations in relation.txt + 1
    if is_kg_list:
        for (entity0, entity1) in seed_pairs:
            graph0 = kg0.subgraph_list_kg[entity0]
            graph1 = kg1.subgraph_list_kg[entity1]

            graph0.x = torch.cat([graph0.x, torch.LongTensor([entity1 + kg1.entity_id_base])])  # global index
            graph1.x = torch.cat([graph1.x, torch.LongTensor([entity0 + kg0.entity_id_base])])  # global index

            # undirected edges
            graph0.edge_index = torch.cat([graph0.edge_index,torch.LongTensor([[graph0.num_size,graph0.y], [graph0.y,graph0.num_size]])], dim=1)
            graph1.edge_index = torch.cat([graph1.edge_index,torch.LongTensor([[graph1.num_size,graph1.y], [graph1.y,graph1.num_size]])], dim=1)

            graph0.edge_attr = torch.cat([graph0.edge_attr,torch.LongTensor([num_relation + kg0.relation_id_base - 1,num_relation + kg0.relation_id_base - 1])])  # global index
            graph1.edge_attr = torch.cat([graph1.edge_attr,torch.LongTensor([num_relation + kg1.relation_id_base - 1,num_relation + kg1.relation_id_base - 1])])  # global index

            graph0.num_size = graph0.num_size + 1
            graph1.num_size = graph1.num_size + 1

            kg0.subgraph_list_kg[entity0] = graph0
            kg1.subgraph_list_kg[entity1] = graph1

    else:
        for (entity0, entity1) in seed_pairs:
            graph0 = kg0.subgraph_list_align[entity0]
            graph1 = kg1.subgraph_list_align[entity1]

            graph0.x = torch.cat([graph0.x, torch.LongTensor([entity1 + kg1.entity_id_base])])  # global index
            graph1.x = torch.cat([graph1.x, torch.LongTensor([entity0 + kg0.entity_id_base])])  # global index

            # undirected edges
            graph0.edge_index = torch.cat([graph0.edge_index, torch.LongTensor([[graph0.num_size, graph0.y], [graph0.y, graph0.num_size]])],dim=1)
            graph1.edge_index = torch.cat([graph1.edge_index, torch.LongTensor([[graph1.num_size, graph1.y], [graph1.y, graph1.num_size]])],dim=1)

            graph0.edge_attr = torch.cat([graph0.edge_attr, torch.LongTensor([num_relation + kg0.relation_id_base - 1, num_relation + kg0.relation_id_base - 1])])  # global index
            graph1.edge_attr = torch.cat([graph1.edge_attr, torch.LongTensor([num_relation + kg1.relation_id_base - 1, num_relation + kg1.relation_id_base - 1])])  # global index
            graph0.num_size = graph0.num_size + 1
            graph1.num_size = graph1.num_size + 1

            kg0.subgraph_list_align[entity0] = graph0
            kg1.subgraph_list_align[entity1] = graph1





