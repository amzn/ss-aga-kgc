import numpy as np
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn



class KnowledgeGraph(nn.Module):
    def __init__(self, lang, kg_train_data, kg_val_data, kg_test_data, num_entity, num_relation, is_supporter_kg,
                 entity_id_base, relation_id_base, device, n_neg_pos=1):
        super(KnowledgeGraph, self).__init__()

        '''
        Store train,test,val data in np form only.
        '''
        self.lang = lang

        self.train_data = kg_train_data  # training set
        self.val_data = kg_val_data
        self.test_data = kg_test_data

        self.entity_id_base = entity_id_base
        self.relation_id_base = relation_id_base

        self.num_relation = num_relation # Total number of relations in relation.txt + 1
        self.num_entity = num_entity
        self.is_supporter_kg = is_supporter_kg

        self.n_neg_pos = n_neg_pos
        self.device = device

        self.subgraph_list_kg = None
        self.subgraph_list_align = None

        self.computed_entity_embedidng_align = None
        self.computed_entity_embedidng_KG = None


        if not is_supporter_kg:
            self.true_tail = self.get_true_tail(self.train_data)
        else:  # also include val data
            self.true_tail = self.get_true_tail(torch.cat([self.train_data, self.val_data], dim=0))  # [numpy]

        # Rewrite to torch form， TODO: r here needs to be global indexing.
        self.h_train, self.r_train, self.t_train = self.train_data[:, 0], self.train_data[:, 1], self.train_data[:, 2]
        self.h_val, self.r_val, self.t_val = self.val_data[:, 0], self.val_data[:, 1], self.val_data[:, 2]
        self.h_test, self.r_test, self.t_test = self.test_data[:, 0], self.test_data[:, 1], self.test_data[:, 2]



    def get_true_tail(self, triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''

        triples_np = triples.numpy()

        true_tail = {}

        for head, relation, tail in triples_np:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)

        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))

        return true_tail



    def generate_batch_data(self,h_all,r_all,t_all,batch_size, shuffle = True):

        h_all = torch.unsqueeze(h_all,dim=1)
        r_all = torch.unsqueeze(r_all, dim=1)
        t_all = torch.unsqueeze(t_all, dim=1)

        #generate negative samples
        total_num = h_all.shape[0]
        t_neg = self.get_negative_samples(total_num).to(h_all.device)

       
        triple_all = torch.cat([h_all,r_all,t_all,t_neg],dim=-1) #[B,3]
        triple_dataloader = DataLoader(triple_all,batch_size = batch_size,shuffle = shuffle)


        return triple_dataloader
  
    def get_negative_samples(self,batch_size_each):

       
        rand_negs = torch.randint(high=self.num_entity, size=(batch_size_each,),
                                  device=self.device)  # [b,num_neg = 1]

        rand_negs = rand_negs.view(-1,1)
        
        return rand_negs



