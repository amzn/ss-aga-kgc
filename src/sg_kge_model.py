# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import src.param as param
import json
import torch
import numpy as np
import torch.nn as nn
from src.gnn import GNN
import torch.nn.functional as F
import math







class SG_KGE(nn.Module):
    def __init__(self,args,entity_bert_emb,relation_bert_emb,kg_node_base_dict,kg_entity_num_dict,device):
        super(SG_KGE, self).__init__()
        '''
        Assume relations are shared across KGs. Otherwise, put the embedding seperately in each KG module.
        '''
        self.args = args
        self.total_num_entity = entity_bert_emb.shape[0]
        self.num_relations = relation_bert_emb.shape[0] + 1 # padding index for self edges of isolated nodes
        self.total_num_relations_unique = self.num_relations 
        self.total_num_kgs = len(kg_node_base_dict)
        self.kg_node_base_dict = kg_node_base_dict
        self.entity_dim = args.entity_dim
        self.relation_dim = args.relation_dim
        self.device = device
        self.criterion = nn.MarginRankingLoss(margin=args.transe_margin, reduction='mean')
        self.kg_entity_num_dict = kg_entity_num_dict
        self.pi = torch.FloatTensor([3.14159265358979323846]).to(self.device)
        
        self.neg_one = torch.LongTensor([-1]).to(device)

        self.entity_embeddings_fixed = nn.Embedding.from_pretrained(torch.FloatTensor(entity_bert_emb),freeze = True)

        if self.args.initialization_method == "bert_w": #Bert + W  (cannot deal with dummy nodes!)

            self.entity_embeddings = self.entity_embeddings_fixed
            self.adapt_ws_entity = nn.Linear(entity_bert_emb.shape[1], self.entity_dim)
            self.adapt_ws_relation = nn.Linear(relation_bert_emb.shape[1], self.relation_dim)
            self.relation_embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(relation_bert_emb), freeze=True)


        elif self.args.initialization_method== "bert_initialized":  # bert initialized, dim = 768 , TODO :not supporting rotate
            self.entity_embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(entity_bert_emb), freeze=False)
            self.relation_embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(relation_bert_emb),
                                                                    freeze=False)


        elif self.args.initialization_method== "random": # random initialized

            self.entity_embeddings = nn.Embedding(self.total_num_entity, self.entity_dim)
            nn.init.xavier_uniform_(self.entity_embeddings.weight)
            self.relation_embeddings = nn.Embedding(self.total_num_relations_unique, self.relation_dim,padding_idx = self.total_num_relations_unique-1)
            nn.init.xavier_uniform_(self.relation_embeddings.weight)


        # Create two GNN encoder
        self.encoder_KG = GNN(in_dim = args.entity_dim, in_edge_dim = args.relation_dim, n_hid = args.encoder_hdim_kg, out_dim = args.entity_dim,
                              n_heads = args.n_heads, n_layers = args.n_layers_KG, dropout = args.dropout)

        self.encoder_align = GNN(in_dim = args.entity_dim, in_edge_dim = args.relation_dim, n_hid = args.encoder_hdim_align, out_dim = args.entity_dim,
                              n_heads = args.n_heads, n_layers = args.n_layers_align, dropout = args.dropout)


        # Initialize relation prior weight
        self.relation_prior = nn.Embedding(self.total_num_relations_unique, 1,padding_idx = self.total_num_relations_unique-1)
        nn.init.xavier_uniform_(self.relation_prior.weight)
        

    def forward_GNN_embedding(self,graph_input, GNN_encoder):
        
        x_features = self.entity_embeddings(graph_input.x) # [num_nodes,d]
        edge_index = graph_input.edge_index
        edge_type_vector = self.relation_prior(graph_input.edge_attr)  # [num_edge]
            
        edge_vector_embedding = self.relation_embeddings(graph_input.edge_attr)
        x_gnn_kg_output = GNN_encoder(x_features, edge_index, edge_type_vector, edge_vector_embedding,graph_input.y,graph_input.num_size)  # [N,d]
        
        return x_gnn_kg_output

    def forward_KG(self,graph_h,kg_batch_each,graph_t,graph_t_neg):
        '''

        :param edge_index:
        :param edge_type:
        :param triple_input:
        :param lang:  To transform local index to the global index
        :return:
        '''
        h = self.forward_GNN_embedding(graph_h,self.encoder_KG) #[num_triple , D]
        r = self.relation_embeddings(kg_batch_each[:,1])
        t = self.forward_GNN_embedding(graph_t,self.encoder_KG) #[num_triple , D]
        t_neg = self.forward_GNN_embedding(graph_t_neg,self.encoder_KG) #[num_triple , D]

        assert h.shape[0] == t.shape[0] == t_neg.shape[0]

        projected_t = self.project_t([h, r])  #[b,d]
        pos_loss = self.define_loss([t, projected_t])  ## [b]
        neg_loss = self.define_loss([t, t_neg])  # [b]
     

        if self.args.knowledge_model == 'rotate':
            # current loss is actually dist
            gm = self.args.rotate_gamma
            pos_loss1 = torch.squeeze(-torch.log(nn.functional.softplus(gm - pos_loss)))
            neg_loss1 = -torch.log(nn.functional.softplus(neg_loss - gm))
            total_loss = (pos_loss1 + neg_loss1) / 2            
            return total_loss.mean()    
        else:
            target = self.neg_one
            total_loss = self.criterion(pos_loss, neg_loss, target)
            
            return total_loss

    def predict(self,h_graph_input,r):
        #(h,r) -> projected t vector

        h = self.forward_GNN_embedding(h_graph_input,self.encoder_KG)

        r = self.relation_embeddings(r)  # todo: check size

        projected_t = self.project_t([h, r])

        return projected_t


    def project_t(self,hr):
        if self.args.knowledge_model == 'transe':
            return hr[0] + hr[1]
        elif self.args.knowledge_model == 'rotate':
            pi = self.pi
            head, relation = hr[0], hr[1] #[b,d]
           
            head = torch.unsqueeze(head,dim=1)
            relation = torch.unsqueeze(relation,dim=1)
            re_head, im_head = torch.chunk(head, 2, dim=2)  # input shape: (None, 1, dim)

            embedding_range = torch.FloatTensor([param.rotate_embedding_range()]).to(self.device)

            # Make phases of relations uniformly distributed in [-pi, pi]
            phase_relation = relation / (embedding_range / pi)

            re_relation = torch.cos(phase_relation)
            im_relation = torch.sin(phase_relation)

            re_tail = re_head * re_relation - im_head * im_relation
            im_tail = re_head * im_relation + im_head * re_relation

            predicted_tail = torch.squeeze(torch.cat([re_tail, im_tail], dim=-1))
            
           
            return predicted_tail


    def forward_align(self,e0_graph,e1_graph):

        e0 = self.forward_GNN_embedding(e0_graph, self.encoder_align)
        e1 = self.forward_GNN_embedding(e1_graph, self.encoder_align)
        
        align_loss = torch.mean(l2distance(e0, e1))  # num batch
        return align_loss



    def extend_seed_align_links(self,lang0, lang1,emb0,emb1, seed_links, device, k_csls = 3):
        """
        Self learning using cross-domain similarity scaling (CSLS) metric for kNN search
        :param kg0: supporter kg
        :param kg1: target kg
        :param seed_links: 2-col np array
        k_csls: how many nodes in neigorhood
        :return:
        TODO: renumber the results
        """

        def cos(v1_bert,v1,v2_bert, v2):
            cos1 = F.cosine_similarity(v1_bert, v2_bert, dim=-1)
            cos2 = F.cosine_similarity(v1, v2, dim=-1)
            
            cos_max = torch.max(cos1,cos2)
            
            return cos_max
        
    
        csls_links = []
        csls_links_renumbered = []

        aligned0 = torch.unique(seed_links[:, 0], return_inverse=False)
        aligned1 = torch.unique(seed_links[:, 1], return_inverse=False)
        
        embedding_matrix0_reshaped = emb0
        embedding_matrix1_reshaped = emb1
        
        entity_index0 = [i + self.kg_node_base_dict[lang0] for i in range(self.kg_entity_num_dict[lang0])]
        entity_index0 = torch.LongTensor(entity_index0).to(device).view(-1,1)
        entity_index1 = [i + self.kg_node_base_dict[lang1] for i in range(self.kg_entity_num_dict[lang1])]
        entity_index1 = torch.LongTensor(entity_index1).to(device).view(-1,1)
        
        emb_bert0 = torch.squeeze(self.entity_embeddings_fixed(entity_index0))
        emb_bert1 = torch.squeeze(self.entity_embeddings_fixed(entity_index1))
        
        kg0_num_entity = embedding_matrix0_reshaped.shape[0]
        kg1_num_entity = embedding_matrix1_reshaped.shape[0]

        # find kNN for each e0
        # mean neighborhood similarity
        e0_neighborhood = torch.zeros([kg0_num_entity, k_csls], dtype=torch.long).to(device)
        e1_neighborhood = torch.zeros([kg1_num_entity, k_csls], dtype=torch.long).to(device)
        e0_neighborhood_cos = torch.zeros(kg0_num_entity).to(device)
        e1_neighborhood_cos = torch.zeros(kg1_num_entity).to(device)

        # find neighborhood
        for e0 in range(kg0_num_entity):
            top_k_from_kg1 = KNN_finder_vec(
                embedding_matrix0_reshaped[e0, :].reshape([1, -1]), embedding_matrix1_reshaped,
                k_csls)  # [array(entity), array(score)]
            neighbood = top_k_from_kg1[0]  # list[entity], possible e1
            e0_neighborhood[e0, :] = neighbood
        for e1 in range(kg1_num_entity):
            top_k_from_kg0 = KNN_finder_vec(
                embedding_matrix1_reshaped[e1, :].reshape([1, -1]), embedding_matrix0_reshaped, k_csls)
            neighbood = top_k_from_kg0[0]  # list[entity], possible e0
            e1_neighborhood[e1, :] = neighbood


        # compute neighborhood similarity
        for e0 in range(kg0_num_entity):
            e0_vec = embedding_matrix0_reshaped[e0,:]
            e0_vec_bert = emb_bert0[e0]
            e0_neighbors = e0_neighborhood[e0, :]  # e0's neighbor in kg1 domain
            neighbor_cos = [cos(e0_vec_bert, e0_vec, emb_bert1[nb,:],embedding_matrix1_reshaped[nb, :] ) for nb in e0_neighbors]
            e0_neighborhood_cos[e0] = torch.mean(torch.FloatTensor(neighbor_cos).to(device))  # r_S

        for e1 in range(kg1_num_entity):
            e1_vec = embedding_matrix1_reshaped[e1,:]
            e1_vec_bert = emb_bert1[e1]
            e1_neighbors = e1_neighborhood[e1, :]  # e0's neighbor in kg1 domain
            neighbor_cos = [cos(emb_bert0[nb,:],embedding_matrix0_reshaped[nb, :],e1_vec_bert, e1_vec) for nb in e1_neighbors]
            e1_neighborhood_cos[e1] = torch.mean(torch.FloatTensor(neighbor_cos).to(device))

        nearest_for_e0 = torch.full((kg0_num_entity, 1), fill_value=-2).to(
            device)  # -2 for not computed, -1 for not found
        nearest_for_e1 = torch.full((kg1_num_entity, 1), fill_value=-2).to(device)

        for true_e0 in range(kg0_num_entity):
            if true_e0 not in aligned0:
                e0_neighbors = e0_neighborhood[true_e0, :]  # e0's neighbor in kg1 domain
                nearest_e1 = torch.LongTensor([-1]).to(device)
                nearest_e1_csls = torch.FloatTensor([-np.inf]).to(device)
                for e1 in e0_neighbors.tolist():
                    if e1 not in aligned1:
                        # rT(Wx_s) is the same for all e1 in e0's neighborhood
                        csls = 2 * cos(emb_bert0[true_e0],embedding_matrix0_reshaped[true_e0, :], emb_bert1[e1],embedding_matrix1_reshaped[e1, :]) - \
                               e1_neighborhood_cos[e1]
                        if csls > nearest_e1_csls:
                            nearest_e1 = e1
                nearest_for_e0[true_e0] = nearest_e1

                # check if they are mutual neighbors
                if nearest_for_e0[true_e0] != torch.LongTensor([-1]).to(device):
                    e1 = nearest_for_e0[true_e0]
                    if nearest_for_e1[e1] == torch.LongTensor([-2]).to(
                            device):  # e1's nearest number not computed yet. compute it now
                        e1_neighbors = e1_neighborhood[e1[0], :]  # e0's neighbor in kg1 domain
                        nearest_e0 = torch.LongTensor([-1]).to(device)
                        nearest_e0_csls = torch.FloatTensor([-np.inf]).to(device)
                        for e0 in e1_neighbors:
                            if e0 not in aligned0:
                                # rT(Wx_s) is the same for all e1 in e0's neighborhood
                                csls = 2 * cos(emb_bert1[e1],embedding_matrix1_reshaped[e1, :], emb_bert0[e0],embedding_matrix0_reshaped[e0, :]) - \
                                       e0_neighborhood_cos[e0]
                                if csls > nearest_e0_csls:
                                    nearest_e0 = e0
                                    nearest_e0_csls = csls
                        nearest_for_e1[e1] = nearest_e0

                    if nearest_for_e1[e1] == true_e0:
                        # mutual e1_neighbors
                        # renumber them to send back to the fused KG.
                        true_e0_renumbered = true_e0+self.kg_node_base_dict[lang0]
                        e1_renumbered = e1 + self.kg_node_base_dict[lang1]
                        csls_links_renumbered.append([true_e0_renumbered, e1_renumbered])
                        csls_links.append(true_e0,e1)

        csls_links = torch.LongTensor(csls_links).to(device)


        return csls_links,csls_links_renumbered



    def define_loss(self,t_true_pred):
        t_true = t_true_pred[0]
        t_pred = t_true_pred[1]
        return torch.norm(t_true - t_pred + 1e-8, dim=1)  # input shape: (None, 1, dim)



    def return_entity_matrix_for_each_kg(self,lang,device):
        #TODO: check!
        node_base = self.kg_node_base_dict[lang]
        entity_num = self.kg_entity_num_dict[lang]

        indexes = torch.LongTensor([i for i in range(node_base,node_base + entity_num)]).to(device)

        entity_embeddings = self.entity_embeddings(indexes)

        return entity_embeddings





def l2distance(a, b):
    # dist = tf.sqrt(tf.reduce_sum(tf.square(a-b), axis=-1))
    dist = torch.norm(a - b + 1e-8, dim=-1)
    return dist


def KNN_finder_vec(input_vec,embedding_matrix,topk ):
    """
    Given a vector, find the kNN entities
    :param num_entity:
    :return:
    """
    #TODO: check caculation!

    predicted_t_vec = torch.squeeze(input_vec)  # shape (batch_size=b, 1, dim) -> (b,dim)
    distance = torch.norm(torch.subtract(embedding_matrix, predicted_t_vec), dim=1) #[b]
    top_k_scores, top_k_t = torch.topk(-distance, k=topk)  # find indices of k largest score. score = neg(distance)

    return [torch.reshape(top_k_t, [1, topk]),
            torch.reshape(top_k_scores, [1, topk])]  # reshape to one row matrix to fit keras model output



def Ranking_all_batch(predicted_t, embedding_matrix):
    """
    kNN finder
    === input: predictor, [input_h_query, input_r_query, embedding_matrix]
    === output:[top_k_entity_idx, top_k_scores(larger is better)] . top_k_entity_idx shape [1,k].
                e.g., [array([[64931, 13553, 20458]]), array([[-1.008282 , -2.0292854, -3.059666]])]
    """
    total_entity = embedding_matrix.shape[0]
    predicted_t = torch.unsqueeze(predicted_t, dim=1) #[b,1,d]
    predicted_t = predicted_t.repeat(1, total_entity, 1) #[b,n,d]

    distance =  torch.norm(predicted_t - embedding_matrix,dim = 2) #[b,n]
    

    top_k_scores, top_k_t = torch.topk(-distance, k=total_entity)




    return top_k_t, top_k_scores

