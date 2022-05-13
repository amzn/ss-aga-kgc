# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from os.path import join
import pandas as pd
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from src.graph_sampler import k_hop_subgraph,subgraph_extractor

from torch_geometric.data import Data
from tqdm import tqdm

from multiprocessing import Pool


class KnowledgeGraph(nn.Module):
    def __init__(self, lang, kg_train_data, kg_val_data, kg_test_data, num_entity, num_relation, is_supporter_kg,entity_id_base,n_neg_pos = 1):
        super(KnowledgeGraph, self).__init__()
        '''
        Store train,test,val data in np form only.
        '''
        self.lang = lang
        self.train_data = kg_train_data  # training set
        self.val_data = kg_val_data
        self.test_data = kg_test_data
        self.entity_id_base = entity_id_base
        self.num_relation = num_relation

        self.num_entity = num_entity
        self.is_supporter_kg = is_supporter_kg
        self.n_neg_pos = n_neg_pos
        
         # Rewrite to torch form
        self.h_train, self.r_train, self.t_train = self.train_data[:,0], self.train_data[:,1], self.train_data[:,2]
        self.h_val, self.r_val, self.t_val = self.val_data[:,0], self.val_data[:,1], self.val_data[:,2]
        self.h_test, self.r_test, self.t_test = self.test_data[:,0], self.test_data[:,1], self.test_data[:,2]
        
        if not is_supporter_kg:
            print(self.lang + " is the targeted KG!")
        else: #also include val data
            self.h_train = torch.cat([self.h_train, self.h_val], axis=0)
            self.r_train = torch.cat([self.r_train, self.r_val], axis=0)
            self.t_train = torch.cat([self.t_train, self.t_val], axis=0)
            print(self.lang + " is the supporter KG!")
        
        self.true_tail = self.get_true_tail() #[numpy]
  
    def generate_batch_data_test_val(self,h_all,r_all,t_all,batch_size, shuffle = False):
        # TODO: once changed into embedding, no need to unsqueeze.
        h_all = torch.unsqueeze(h_all,dim=1)
        r_all = torch.unsqueeze(r_all, dim=1)
        t_all = torch.unsqueeze(t_all, dim=1)
 
        triple_all = torch.cat([h_all,r_all,t_all],dim=-1) #[N,4]
        triple_dataloader = DataLoader(triple_all,batch_size = batch_size,shuffle = shuffle)
        return triple_dataloader   
    
    def generate_batch_data(self,h_all,r_all,t_all,batch_size, shuffle = True):
        # TODO: once changed into embedding, no need to unsqueeze.
       
        h_all = torch.unsqueeze(h_all,dim=1)
        r_all = torch.unsqueeze(r_all, dim=1)
        t_all = torch.unsqueeze(t_all, dim=1)
 
        #1. create t_ng for training
  
        t_neg_all = self.get_negative_sample(self.n_neg_pos,h_all.numpy(),r_all.numpy())
        

                
        tail_negs = torch.cat(t_neg_all,dim = 0) #[N,num_neg]
        tail_negs = torch.unsqueeze(tail_negs,dim = 1)
        
          
        # 2. create 4-tuple
        triple_all = torch.cat([h_all,r_all,t_all,tail_negs],dim=-1) #[N,4]
        triple_dataloader = DataLoader(triple_all,batch_size = batch_size,shuffle = shuffle)
        
        return triple_dataloader
       
    def get_true_tail(self):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''  
    
        heads = self.h_train.numpy()
        relations = self.r_train.numpy()
        tails = self.t_train.numpy()
        
            
        true_tail = {}
        legnths = []
        for i,head in enumerate(heads):
            relation = relations[i]
            tail = tails[i]
            if (head, relation) not in true_tail:
                true_tail[(int(head), int(relation))] = [-1]
            true_tail[(int(head), int(relation))].append(tail)

        for (head, relation) in true_tail.keys():
            tmp = np.array(list(set(true_tail[(head, relation)])))   
            true_tail[(head, relation)] = tmp
            legnths.append(tmp.shape[0])
            
      
        return true_tail

    def get_negative_sample(self, n_neg_pos,heads,relations):
        
        total_sample = heads.shape[0]
        
        neg_all = []
        for i in range(total_sample):
            head = int(heads[i])
            relation = int(relations[i])
           
            
            
            negative_sample_list = []
            negative_sample_size = 0
            while negative_sample_size < n_neg_pos:
                negative_sample = np.random.randint(self.num_entity, size=n_neg_pos*2)
                mask = np.in1d(
                    negative_sample, 
                    self.true_tail[(head, relation)], 
                    assume_unique=True, 
                    invert=True)
           
                negative_sample = negative_sample[mask]
                negative_sample_list.append(negative_sample)
                negative_sample_size += negative_sample.size
        
            negative_sample = np.concatenate(negative_sample_list)[:n_neg_pos]
            negative_sample = torch.LongTensor(negative_sample)
            
            neg_all.append(negative_sample)
        
        return neg_all
    

class KG_triple_loader(Dataset):
    """Dataset implementation for handling FB15K and FB15K-237."""

    def __init__(self, triples):
        self.data = triples

    def __len__(self):
        """Denotes the total number of samples."""
        return len(self.data)

    def __getitem__(self, index):
        """Returns (head id, relation id, tail id)."""
        samples = self.data[index]

        return torch.LongTensor(samples)


class ParseData(object):
    def __init__(self, data_path = 'data/', target_kg = 'el', is_unified = False):
        self.data_path = data_path + "/"
        self.data_entity = self.data_path + "entity/"
        self.data_kg = self.data_path + "kg/"
        self.data_align = self.data_path + "seed_alignlinks/"
        self.is_unified = is_unified   # only used when generating the initial graph (weather want to incorporate their val data as well into training)
        self.target_kg = target_kg

        self.kg_node_base_dict, self.kg_list, self.kg_entity_num_dict, self.total_entity_num= self.get_kg_node_base_dict()
        self.num_relation_kg = self.get_num_relations() + 1

    def load_data(self,args,num_hop,k):
        '''
        :return:
        1. X (bert embedding matrix), R (bert embedding matrix)
        2. Seed alignment (preserved,masked,whole)
        3. G_initial ( [2,num_edges]),
        4. list of KG object
        5. kg_node_base_dict
        '''
        relation_bert_emb = np.load(self.data_path + "relation_embedding_title.npy")
        if args.entity_emb  == "title":
            entity_bert_emb = np.load(self.data_path + "entity_embedding_title.npy")
        else:
            entity_bert_emb = np.load(self.data_path + "entity_embedding_description.npy")

        # normalize features to be within [-1,1]
        relation_bert_emb = self.normalize_fature(relation_bert_emb)
        entity_bert_emb = self.normalize_fature(entity_bert_emb)

#         assert relation_bert_emb.shape[0] == self.num_relation_kg + 1
#         assert entity_bert_emb.shape[0] == sum(self.kg_entity_num_dict.values())

        seeds_masked, seeds_all, seeds_preserved = self.load_all_to_all_seed_align_links(args.preserved_percentage)

        # TODO :seeds_all will be used for testing
        edge_index, edge_type= self.generate_whole_graph_initial(seeds_preserved)
        
        print("Total number of edges:%d" % edge_index.shape[1])

        kg_object_dict = self.create_KG_object()
        
        # Get subgraph_list
        total_relation_embedding_num = self.num_relation_kg 
        sub_graph_list = create_subgraph_list(edge_index,edge_type,self.total_entity_num,total_relation_embedding_num,num_hop,k)
    
        return edge_index,edge_type, entity_bert_emb, relation_bert_emb, seeds_masked, seeds_all, seeds_preserved, kg_object_dict, self.kg_node_base_dict, self.kg_entity_num_dict,self.num_relation_kg,sub_graph_list

    
    def normalize_fature(self,input_embedding):
        input_max = input_embedding.max()
        input_min = input_embedding.min()

        # Normalize to [-1, 1]
        input_embedding_normalized = (input_embedding - input_min) * 2 / (input_max - input_min) - 1

        return input_embedding_normalized


    def get_kg_node_base_dict(self):
        
        entity_files = list(os.listdir(self.data_entity))
        entity_files = sorted(list(filter(lambda x: x[-3:] == "tsv",entity_files)))
        entity_files = list(entity_files)

        kg_node_base_dict = {}
        kg_entity_num = []
        kg_entity_num_dict = {}
        langs = []

        for entity_file in entity_files:
            f = open(self.data_entity + entity_file, 'r')
            lines = f.readlines()
            f.close()

            kg_name = entity_file[:2]
            langs.append(kg_name)
            kg_entity_num.append(len(lines))
            kg_entity_num_dict[kg_name] = len(lines)

        for i, each_lang in enumerate(langs):
            if i == 0:
                kg_node_base_dict[each_lang] = 0
            else:
                kg_node_base_dict[each_lang] = np.sum(kg_entity_num[:i])

        return kg_node_base_dict,langs,kg_entity_num_dict, np.sum(kg_entity_num)

    
    def get_num_relations(self):

        f = open(self.data_path + "relations.txt",'r')
        lines = f.readlines()
        f.close()

        return len(lines)

    def load_all_to_all_seed_align_links(self, preserved_percentage=0.7):
        seeds_preserved = {}  # { (lang1, lang2): 2-col np.array }
        seeds_masked = {}
        seeds_all = {}
        for f in os.listdir(self.data_align):  # e.g. 'el-en.tsv'
            lang1 = f[0:2]
            lang2 = f[3:5]
            links = pd.read_csv(join(self.data_align, f), sep='\t',header = None).values.astype(int)  # [N,2] ndarray

            total_link_num = links.shape[0]
            preserved_idx = list(sorted(
                np.random.choice(np.arange(total_link_num), int(total_link_num * preserved_percentage), replace=False)))
            masked_idx = list(filter(lambda x: x not in preserved_idx, np.arange(total_link_num)))

            assert len(masked_idx) + len(preserved_idx) == total_link_num

            preserved_links = links[preserved_idx, :]
            masked_links = links[masked_idx, :]

            seeds_masked[(lang1, lang2)] = torch.LongTensor(masked_links)
            seeds_all[(lang1, lang2)] = torch.LongTensor(links)
            seeds_preserved[(lang1, lang2)] = torch.LongTensor(preserved_links)  # to be used to generate the whole graph

        return seeds_masked, seeds_all, seeds_preserved

    def generate_whole_graph_initial(self, seeds_aligned):
        '''
        seeds_aligned: dict[{lang1,lang2}]= torhc.LongTensor
        Called for the initial graph construction
        :param aligned_seeds: np.array (whole , masked ,preserved)
        :return:
        '''
        edge_index_all = []
        edge_weight_all = []


        # for IN-KG links
        for kg_name in self.kg_list:
            kg_node_base = self.kg_node_base_dict[kg_name]
            if kg_name == self.target_kg:
                is_target = True
            else:
                is_target = False

            edge_index_each, edge_weight_each = get_kg_edges_for_each(self.data_kg, kg_name, kg_node_base, is_target,
                                                                      is_unified=False)

            edge_index_all.append(edge_index_each)
            edge_weight_all.append(edge_weight_each)
        
       

        # for cross KG links
        for (lang1, lang2), align_link_each in seeds_aligned.items():
            edge_index_each, edge_weight_each = get_align_edges_for_each(seeds_aligned, lang1, lang2, self.kg_node_base_dict,
                                                                         self.num_relation_kg)

            edge_index_all.append(edge_index_each)
            edge_weight_all.append(edge_weight_each)


        edge_index = np.concatenate(edge_index_all, axis=1)  # [2,num_edges]
        edge_weight = np.concatenate(edge_weight_all, axis=0)  # [num_edges]
        
        edge_index = torch.LongTensor(edge_index)
        edge_weight = torch.LongTensor(edge_weight)
        
 
      
        return edge_index, edge_weight


    def create_KG_object(self):
        # INDEX ONLY!
        kg_objects_dict = {}
        for lang in self.kg_list:
            kg_train_data, kg_val_data, kg_test_data = load_data_each_kg(self.data_kg, lang)  # use suffix 1 for supporter kg, 0 for target kg

            if lang == self.target_kg:
                is_supporter_kg = False
            else:
                is_supporter_kg = True
            
            kg_each = KnowledgeGraph(lang, kg_train_data, kg_val_data, kg_test_data, self.kg_entity_num_dict[lang], self.num_relation_kg,
                                    is_supporter_kg,self.kg_node_base_dict[lang])
            kg_objects_dict[lang] = kg_each
        return kg_objects_dict



def load_data_each_kg(data_dir, language):
    """
    :return: triples (n_triple, 3) np.int np.array
    :param testfile_suffix: '-val.tsv' or '-test.tsv'. Default '-val.tsv'
    """

    train_df = pd.read_csv(join(data_dir, language + '-train.tsv'), sep='\t', header=None, names=['v1', 'relation', 'v2'])
    val_df = pd.read_csv(join(data_dir, language + '-val.tsv'), sep='\t', header=None, names=['v1', 'relation', 'v2'])
    test_df = pd.read_csv(join(data_dir, language + '-test.tsv'), sep='\t', header=None, names=['v1', 'relation', 'v2'])


    triples_train = train_df.values.astype(np.int)
    triples_val = val_df.values.astype(np.int)
    triples_test = test_df.values.astype(np.int)
    return KG_triple_loader(triples_train), KG_triple_loader(triples_val), KG_triple_loader(triples_test)


def get_align_edges_for_each(seeds_aligned,lang1,lang2, kg_node_base_dict,num_relation):
    '''
    TODO : tensor computation !! Now it is numpy computaton
    Num_relation is the total num including the alignment type!!!!!

    :param seeds_aligned:
    :param lang1:
    :param lang2:
    :param kg_node_base_dict:
    :param num_relation:
    :return:
    '''
    aligned_links = seeds_aligned[(lang1,lang2)]

    lang1_base = kg_node_base_dict[lang1]
    lang2_base = kg_node_base_dict[lang2]

    senders = aligned_links[:, 0] + lang1_base
    receivers = aligned_links[:, 1] + lang2_base


    sender_list = []
    receiver_list = []


    sender_list += senders.tolist()
    sender_list += receivers.tolist()

    receiver_list += receivers.tolist()
    receiver_list += senders.tolist()


    # alignment edge is viewed as a new relation type! index = num_of_relation
    weight_list = [num_relation-1 for _ in range(len(receiver_list))]

    edge_index = np.vstack((sender_list, receiver_list))
    edge_weight = np.asarray(weight_list)

    return edge_index, edge_weight


def get_kg_edges_for_each(data_dir,language,node_index_base,is_target_KG = False, testfile_suffix = '-val.tsv',is_unified = False):

    '''
    Note: generate undirected graph! (bidirectional)
    :param data_dir:
    :param language:
    :param testfile_suffix:
    :param is_unified: weather incorporate supporter KG's validation data to construct the graph
    :return: 1. edge_index [2,num_edge]: edges including cross-time
             2. edge_weight [num_edge]: edge weights
    '''

    train_df = pd.read_csv(join(data_dir, language + '-train.tsv'), sep='\t', header=None,
                           names=['v1', 'relation', 'v2'])

    val_df = pd.read_csv(join(data_dir, language + testfile_suffix), sep='\t', header=None,
                         names=['v1', 'relation', 'v2'])

    # Training data graph construction
    sender_node_list = (train_df['v1'].values.astype(np.int) + node_index_base).tolist()
    sender_node_list += (train_df['v2'].values.astype(np.int) + node_index_base).tolist()

    receiver_node_list = (train_df['v2'].values.astype(np.int) + node_index_base).tolist()
    receiver_node_list += (train_df['v1'].values.astype(np.int) + node_index_base).tolist()


    edge_weight_list =  train_df['relation'].values.astype(np.int).tolist() +  train_df['relation'].values.astype(np.int).tolist()

    # unified: Adding validation edges from supporter KG as well
    if (not is_unified) and (not is_target_KG):
        sender_node_list += (val_df['v1'].values.astype(np.int) + node_index_base).tolist()
        sender_node_list += (val_df['v2'].values.astype(np.int) + node_index_base).tolist()

        receiver_node_list += (val_df['v2'].values.astype(np.int) + node_index_base).tolist()
        receiver_node_list += (val_df['v1'].values.astype(np.int) + node_index_base).tolist()

        edge_weight_list += val_df['relation'].values.astype(np.int).tolist()
        edge_weight_list += val_df['relation'].values.astype(np.int).tolist()


    edge_index = np.vstack((sender_node_list, receiver_node_list))
    edge_weight = np.asarray(edge_weight_list)

#     print(language + "in_kg:" + str(edge_index.shape[1]))
    return edge_index, edge_weight





    
    


def create_subgraph_list(edge_index,edge_value,total_num_nodes,total_num_edges,num_hops,k):
    
    # Adding self-loop for nodes without edges
    # TODO padding self=edges for those do not have edges
    sub_graph_list = []
    num_edges = []
    zero_edge_num = 0
 
    node_list = [i for i in range(total_num_nodes)]
    

    for i in tqdm(node_list):
       
        [tmp1,tmp2,tmp3,tmp4] = k_hop_subgraph([i],num_hops,edge_index,num_nodes=total_num_nodes,relabel_nodes=True)
        x = tmp1 
        edge_index_each = tmp2[:,:k]
        # edge value can be zero!!!!!!!!!!!!!
        edge_value_masked = (edge_value + 1)*tmp4
        edge_attr = edge_value_masked[edge_value_masked.nonzero(as_tuple=True)] - 1
        edge_attr = edge_attr[:k]
        
        assert edge_attr.shape[0] == edge_index_each.shape[1]
        
        if edge_index_each.shape[1] == 0: # sadding self_edge
            zero_edge_num += 1
            edge_index_each = torch.LongTensor([[0],[0]]) 
            edge_attr = torch.LongTensor([total_num_edges])
       
        
        node_position = torch.LongTensor([tmp3])
        num_size = len(tmp1)
        num_size = torch.LongTensor([num_size])
        graph_each = Data(x=x, edge_index=edge_index_each, edge_attr=edge_attr, y = node_position,num_size = num_size)
      
        sub_graph_list.append(graph_each)
        
        num_edges.append(edge_index_each.shape[1])
                   
    
    print("Average subgraph edges %.2f" % np.mean(num_edges))
    print("Num of no edge %d" % zero_edge_num)     
    
    
#     # Check whether have zero edges!
#     num_empty = 0
#     for each_graph in sub_graph_list:
#         if each_graph.edge_index.shape[1] == 0:
#             num_empty += 1
            
#         max_node_index = torch.max(each_graph.x).item()
#         max_node_index2 = torch.max(each_graph.edge_index).item()
#         max_edge_index = torch.max(each_graph.edge_attr).item()
        
#         assert max_node_index<total_num_nodes
#         assert max_node_index2<total_num_nodes
#         assert max_edge_index<=total_num_edges
# #     print("After processing, num empty is %d" % num_empty)

    return sub_graph_list













