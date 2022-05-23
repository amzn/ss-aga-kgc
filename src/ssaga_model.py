import torch
import torch.nn as nn
from src.gnn import GNN
from src.utils import nodes_to_graph, Ranking_all_batch,subgrarph_list_from_alignment
import torch.nn.functional as F
import numpy as np


def l2distance(a, b):
    # dist = tf.sqrt(tf.reduce_sum(tf.square(a-b), axis=-1))
    dist = torch.norm(a - b + 1e-8, dim=-1)
    return dist



def KNN_distance_batch(query_embedding, candidate_embedding):
    '''
    :param query_embedding: [N1,d]
    :param candidate_embedding: [N2,d]
    :return:
    '''

    query_embedding = torch.unsqueeze(query_embedding, dim=1)  # [N1,1,d]
    candidate_embedding = torch.unsqueeze(candidate_embedding, dim=0)  # [1, N2,d]
    distance = torch.norm(query_embedding - candidate_embedding, dim=2)  #[N1,N2]
    return distance




def get_KNN_batch(query_original_and_bert, embedding_matrix_original_and_bert, k,device,batch_size):
    '''
    TODO: batch computing!!!
    :param query_original_and_cos: ([N1,d], [N1,d])
    :param embedding_matrix_original_and_cos:  ([N2,d], [N2,d])
    :param k:
    :return: [N1,k]
    '''
    query_emb0, query_emb1 = torch.split(query_original_and_bert[0],batch_size,dim=0), torch.split(query_original_and_bert[1],batch_size,dim=0)
    candidate_emb0, candidate_emb1 = embedding_matrix_original_and_bert[0], embedding_matrix_original_and_bert[1]

    distance_table = []
    total_num_batch = len(query_emb0)
    for i in range(total_num_batch):
        distance_0 = KNN_distance_batch(query_emb0[i], candidate_emb0)
        distance_table.append(distance_0)

    distance_table = torch.cat(distance_table, dim=0)
    _, top_k_indexes  = torch.topk(-distance_table, k=k)

    return top_k_indexes.type(torch.LongTensor).to(device)

def get_neighbor_embeddings(neighbor_indexes, embeddings, embeddings_bert):
    '''

    :param neighbor_indexes: [N_node_query, k_neighbots], neighbor id for each node
    :param embeddings: [N_all,d]
    :param embeddings_bert: [N_all,d]
    :return: [N_node_query, k_neighbots, d]
    '''
    num_k = neighbor_indexes.shape[1]
    neighbor_indexes_flatten = torch.reshape(neighbor_indexes,(-1,))  # [N_query* k_neighbors]

    neighbor_embeddings_flatten = torch.index_select(embeddings,0,neighbor_indexes_flatten) # [N_query* k_neighbors,d]
    neighbor_embeddings_splited = torch.split(neighbor_embeddings_flatten,num_k,dim=0) # N_query* [k_neighbors,d]
    neighbor_embeddings_true = torch.stack(neighbor_embeddings_splited,dim=0) #[N_node_query, k_neighbots, d]

    neighbor_embeddings_flatten_bert = torch.index_select(embeddings_bert, 0,neighbor_indexes_flatten.cpu().detach())  # [N_query* k_neighbors,d]
    neighbor_embeddings_splited_bert = torch.split(neighbor_embeddings_flatten_bert, num_k, dim=0)  # N_query* [k_neighbors,d]
    neighbor_embeddings_true_bert = torch.stack(neighbor_embeddings_splited_bert, dim=0)  # [N_node_query, k_neighbots, d]

    return (neighbor_embeddings_true,neighbor_embeddings_true_bert)



def compute_cosine_batch(k_neighbor_embedding,query_node_embedding):

    '''
    TODO: add bert embeddings into it.
    :param k_neighbor_embedding: [N_query_node, k, d],  [N_query_node, k, d]
    :param query_node_embedding: [N_query_node,d], [N_query_node,d]
    :return:
    '''
    k_neighbor_embedding_original,k_neighbor_embedding_bert = k_neighbor_embedding[0],k_neighbor_embedding[1]
    query_node_embedding_original,query_node_embedding_bert = query_node_embedding[0],query_node_embedding[1]

    query_node_embedding_original = torch.unsqueeze(query_node_embedding_original,dim=1) #[N,1,D]
    cosine_table_original = F.cosine_similarity(k_neighbor_embedding_original, query_node_embedding_original, dim=-1) #[N_query_node, k]

    query_node_embedding_bert = torch.unsqueeze(query_node_embedding_bert, dim=1)  # [N,1,D]
    cosine_table_bert = F.cosine_similarity(k_neighbor_embedding_bert, query_node_embedding_bert,dim=-1).to(query_node_embedding_original.device)  # [N_query_node, k]


    cosine_table = torch.where(cosine_table_original>cosine_table_bert,cosine_table_original,cosine_table_bert)

    return cosine_table, torch.mean(cosine_table,dim=-1)



class SSAGA(nn.Module):
    def __init__(self, args, entity_bert_emb,num_relations, num_entities, num_KGs):
        super(SSAGA, self).__init__()
        '''
        Assume relations are shared across KGs. Otherwise, put the embedding seperately in each KG module.
        '''
        self.args = args
        self.batch_size = args.batch_size
        self.num_KGs = num_KGs
        self.total_num_entity = num_entities
        self.entity_bert_emb = torch.FloatTensor(entity_bert_emb)

        assert entity_bert_emb.shape[0] == num_entities

        self.num_relations = num_relations

        self.entity_dim = args.entity_dim
        self.relation_dim = args.relation_dim
        self.device = args.device
        self.criterion_KG = nn.MarginRankingLoss(margin=args.transe_margin, reduction='mean')
        self.criterion_align = nn.MarginRankingLoss(margin=args.align_margin, reduction='mean')


        # 1. Embedding initialization

        self.entity_embedding_layer = nn.Embedding(self.total_num_entity, self.entity_dim)
        nn.init.xavier_uniform_(self.entity_embedding_layer.weight)

        self.rel_embedding_layer = nn.Embedding(self.num_relations,self.relation_dim)
        nn.init.xavier_uniform_(self.rel_embedding_layer.weight)

        self.relation_prior = nn.Embedding(self.num_relations, 1)
        nn.init.xavier_uniform_(self.relation_prior.weight)

        # 2. Create two GNN encoder
        self.encoder_KG = GNN(in_dim=args.entity_dim, in_edge_dim=args.relation_dim, n_hid=args.encoder_hdim_kg,
                              out_dim=args.entity_dim,
                              n_heads=args.n_heads, n_layers=args.n_layers_KG, dropout=args.dropout)

        self.encoder_align = GNN(in_dim=args.entity_dim, in_edge_dim=args.relation_dim, n_hid=args.encoder_hdim_align,
                                 out_dim=args.entity_dim,
                                 n_heads=args.n_heads, n_layers=args.n_layers_align, dropout=args.dropout)

    def forward_GNN_embedding(self, graph_input, GNN):
        # Original GNN implementation
        x_features = self.entity_embedding_layer(graph_input.x)  # [num_nodes,d]
        edge_index = graph_input.edge_index
        edge_type_vector = self.relation_prior(graph_input.edge_attr)  # [num_edge]

        edge_vector_embedding = self.rel_embedding_layer(graph_input.edge_attr)
        x_gnn_kg_output = GNN(x_features, edge_index, edge_type_vector, edge_vector_embedding,
                                      graph_input.y, graph_input.num_size)  # [N,d]

        return x_gnn_kg_output

    def forward_kg(self, h_graph, sample, t_graph, t_neg_graph):
        '''

        :param h_graph:
        :param sample:
        :param t_graph:
        :param t_neg_graph:
        :param GNN:
        :return:
        '''

        h = self.forward_GNN_embedding(h_graph, self.encoder_KG).unsqueeze(1)  #[b,1,D]

        r = self.rel_embedding_layer(sample[:, 1]).unsqueeze(1)  #[b,1,D]

        t = self.forward_GNN_embedding(t_graph, self.encoder_KG).unsqueeze(1)  #[b,1,D]

        t_neg = self.forward_GNN_embedding(t_neg_graph, self.encoder_KG).unsqueeze(1)

        projected_t = self.project_t([h, r])  ##
        pos_loss = self.define_loss([t, projected_t])  ## [b,1]

        #         batch_size_each = sample.size()[0]
        #         t_neg = self.get_negative_samples(batch_size_each)

        neg_losses = self.define_loss([t, t_neg])  # [b,num_neg]

        # TransE
        neg_loss = torch.mean(neg_losses, dim=-1)

        target = torch.tensor([-1], dtype=torch.long, device=self.device)
        total_loss = self.criterion_KG(pos_loss, neg_loss, target)

        return total_loss

    def project_t(self, hr):
        '''
        hr: embeddings of head and relation
        :param hr:
        :return:
        '''
        return hr[0] + hr[1]


    def predict(self, h_emb, r):
        # Support batching.
        # (h,r) (r is index_only) -> projected t vector
        entity_dim = h_emb.shape[1]
        h = h_emb.view(-1, entity_dim).unsqueeze(1)
        r = self.rel_embedding_layer(r).unsqueeze(1)
        projected_t = self.project_t([h, r])

        return projected_t

    def define_loss(self, t_true_pred):
        t_true = t_true_pred[0]
        t_pred = t_true_pred[1]
        return torch.norm(t_true - t_pred + 1e-8, dim=2)  # input shape: (None, 1, dim)

    def forward_align(self, e0_graph, e1_graph,e1_graph_neg):
        e0 = self.forward_GNN_embedding(e0_graph, self.encoder_align)
        e1 = self.forward_GNN_embedding(e1_graph, self.encoder_align)
        align_loss = torch.mean(l2distance(e0, e1))  # num batch

        return align_loss

    def forward_align_RANKING(self, e0_graph, e1_graph, e1_graph_neg):

        e0 = self.forward_GNN_embedding(e0_graph, self.encoder_align)
        e1 = self.forward_GNN_embedding(e1_graph, self.encoder_align)
        e1_neg = self.forward_GNN_embedding(e1_graph_neg, self.encoder_align)
        align_loss_pos = l2distance(e0, e1)  # num batch
        align_loss_neg = l2distance(e0,e1_neg)
        target = torch.tensor([-1],dtype=torch.long,device=self.device)
        align_loss = self.criterion_align(align_loss_pos,align_loss_neg,target)
        return align_loss

    def forward_align_new_multiple_neg(self, e0_graph, e1_graph, e1_graph_neg):

        e0 = self.forward_GNN_embedding(e0_graph, self.encoder_align)
        e1 = self.forward_GNN_embedding(e1_graph, self.encoder_align)
        e1_neg = self.forward_GNN_embedding(e1_graph_neg, self.encoder_align)  #[b*n_neg,d]

        align_loss_pos = l2distance(e0, e1)  # num batch  [b]
        align_loss_neg = l2distance(e0,e1_neg)  #[b,n_neg]
        align_loss_neg = torch.mean(align_loss_neg,dim=-1)
        target = torch.tensor([-1],dtype=torch.long,device=self.device)
        align_loss = self.criterion_align(align_loss_pos,align_loss_neg,target)
        return align_loss

  

    def get_kg_embeddings_matrix(self,kg,batch_size,device,is_kg = True):
        '''
        Compute the entity_embedding matrixes in advance, based on self.encoder_GNN/ align_GNN
        :param kg:
        :param batch_size:
        :param device:
        :return:
        '''
        with torch.no_grad():
            node_index_tensor = torch.LongTensor([i for i in range(kg.num_entity)])
            graphs = nodes_to_graph(kg.subgraph_list_kg, node_index_tensor, batch_size)

            embedding_list = []
            if is_kg:
                for graph_batch in graphs:
                    assert graph_batch.edge_index.shape[1] == graph_batch.edge_attr.shape[0]
                    graph_batch = graph_batch.to(device)  # only used to retrive relations
                    node_embeddings = self.forward_GNN_embedding(graph_batch, self.encoder_KG)
                    embedding_list.append(node_embeddings)
            else:
                for graph_batch in graphs:
                    assert graph_batch.edge_index.shape[1] == graph_batch.edge_attr.shape[0]
                    graph_batch = graph_batch.to(device)  # only used to retrive relations
                    node_embeddings = self.forward_GNN_embedding(graph_batch, self.encoder_align)
                    embedding_list.append(node_embeddings)

            embedding_table = torch.cat(embedding_list, dim=0).to(device)  # [n,d]

        return embedding_table


    def extend_seed_align_links(self, kg0, kg1, seeds, device, k_csls=3):
        """
        Self learning using cross-domain similarity scaling (CSLS) metric for kNN search
        :param kg0: supporter kg
        :param kg1: target kg
        :param seed_links: 2-col np array
        k_csls: how many nodes in neigorhood
        :return:
        TODO: renumber the results
        TODO: avoid redundant computation.
        """

        csls_links = []

        aligned0_entities = torch.unique(seeds[:, 0], return_inverse=False).to(device)
        aligned1_entities = torch.unique(seeds[:, 1], return_inverse=False).to(device)

        # avoid redundant computation within one epoch. TODO: need to empty computed_entity_embeddings_align after all alignment generations in one epoch.
        if kg0.computed_entity_embedidng_align == None:
            kg0.computed_entity_embedidng_align = self.get_kg_embeddings_matrix(kg0, self.args.batch_size,
                                                                                self.args.device, is_kg=False)

        if kg1.computed_entity_embedidng_align == None:
            kg1.computed_entity_embedidng_align = self.get_kg_embeddings_matrix(kg1, self.args.batch_size,
                                                                                self.args.device, is_kg=False)

        embedding_matrix0 = kg0.computed_entity_embedidng_align
        embedding_matrix1 = kg1.computed_entity_embedidng_align


        entity_index0 = torch.LongTensor([i + kg0.entity_id_base for i in range(kg0.num_entity)]).view(-1)
        entity_index1 = torch.LongTensor([i + kg1.entity_id_base for i in range(kg1.num_entity)]).view(-1)

        emb_bert0 = torch.index_select(self.entity_bert_emb, 0, entity_index0)
        emb_bert1 = torch.index_select(self.entity_bert_emb, 0, entity_index1)



        # Step1 : find KNN for each e0:using batch: TODO: try not directly compute the whole cosine table.
        e0_neighborhood_indexes = get_KNN_batch((embedding_matrix0, emb_bert0), (embedding_matrix1, emb_bert1),k=k_csls, device=self.device,batch_size=self.batch_size)  # [N_0, k_csls], [N0,N1]
        e1_neighborhood_indexes = get_KNN_batch((embedding_matrix1, emb_bert1), (embedding_matrix0, emb_bert0),k=k_csls, device=self.device,batch_size=self.batch_size)  # [N_0, k_csls], [N0,N1]


        # Step2 : compute mean neighborhood cosine similarity: using batch  TODO: use the precomputed cos_table to compute mean
        e0_neighborhood_embeddings_and_bert = get_neighbor_embeddings(e0_neighborhood_indexes,embedding_matrix1,emb_bert1)  # [N_0, k_csls, d] #CPU
        e1_neighborhood_embeddings_and_bert = get_neighbor_embeddings(e1_neighborhood_indexes,embedding_matrix0,emb_bert0)  # [N_1, k_csls, d] #CPU


        e0_neighborhood_cos_table,e0_neighborhood_cos = compute_cosine_batch(e0_neighborhood_embeddings_and_bert, (embedding_matrix0,emb_bert0))  # [N_0,k], [N_0]
        e1_neighborhood_cos_table,e1_neighborhood_cos = compute_cosine_batch(e1_neighborhood_embeddings_and_bert, (embedding_matrix1,emb_bert1))  # [N_1,k], [N_1] # TODO: whether the second table needs to be computed here.


        # Step3 : find mutual nearest neighbors.compute CSLS, and decide whether mutual nearest neighbors. assume each node should have only at most one alignment in each KG.
        # 1.) Filter pairs by checking whther 1. e0 in aligned0 or 2. e1_list in aligned1
        nearest_for_e0 = torch.full((kg0.num_entity, 1), fill_value=-2, dtype=torch.long).to(device)  # -2 for not computed, -1 for not found
        nearest_for_e1 = torch.full((kg1.num_entity, 1), fill_value=-2, dtype=torch.long).to(device)  # -2 for not computed, -1 for not found
        # 1.) Filter by e0: filter out e0s that are already paired
        e0_candidate_indexes = []
        for i in torch.arange(kg0.num_entity).to(device):
            if i not in aligned0_entities:
                e0_candidate_indexes.append(i)

        e0_candidate_indexes = torch.LongTensor(e0_candidate_indexes).to(device)  # [N_0_filtered]
        e0_neighborhood_indexes_filtered = torch.index_select(e0_neighborhood_indexes, 0, e0_candidate_indexes)  # [N_0_filtered,k_csls]

        # 2.）Filter by e1: filter out e1s that are already paird, remove e0 that has zero e1s.
        def get_e1_mask(x):
            if x in aligned1_entities:
                return x
            else:
                return -1

        e0_neighborhood_indexes_filtered_e1 = e0_neighborhood_indexes_filtered.cpu().apply_(lambda x: get_e1_mask(x)).to(device)  # -1 to denote not valid e1, #[N_0_filtered,k_csls]
        e0_candidate_neighbor_sum = torch.sum(e0_neighborhood_indexes_filtered_e1,dim=1)  # [N_0_filtered], if ==-3 ----> no candidate.

        for i, true_e0 in enumerate(e0_candidate_indexes):
            if e0_candidate_neighbor_sum[i] == -k_csls:  # no aligned pairs, all -1
                continue

            nearest_e1 = -1
            nearest_e1_csls = -np.inf
            e1_candiate_list = e0_neighborhood_indexes_filtered_e1[i]  # [k_csls]
            # 1. compute csls for all candidates
            for j,e1_candidate in enumerate(e1_candiate_list):
                # rT(Wx_s) is the same for all e1 in e0's neighborhood
                if e1_candidate != -1:  # Todo: check type, check device
                    csls = 2 * e0_neighborhood_cos_table[true_e0][j] - e1_neighborhood_cos[e1_candidate]
                    if csls > nearest_e1_csls:
                        nearest_e1 = e1_candidate
                        nearest_e1_csls = csls
            nearest_for_e0[true_e0] = nearest_e1

            # 2. check whether mutual csls
            if nearest_for_e0[true_e0] != -1:
                e1 = nearest_for_e0[true_e0].type(torch.LongTensor)
                if nearest_for_e1[e1] == -2:  # e1's nearest number not computed yet. compute it now
                    e1_neighbors = e1_neighborhood_indexes[e1]  # e0's neighbor in kg1 domain
                    nearest_e0 = -1
                    nearest_e0_csls = -np.inf
                    for q,e0 in enumerate(e1_neighbors[0]):
                        if e0 not in aligned0_entities:
                            # rT(Wx_s) is the same for all e1 in e0's neighborhood
                            csls = 2 * e1_neighborhood_cos_table[e1][0][q]- e0_neighborhood_cos[e0]
                            if csls > nearest_e0_csls:
                                nearest_e0 = e0
                                nearest_e0_csls = csls
                    nearest_for_e1[e1] = nearest_e0

                if nearest_for_e1[e1] == true_e0:
                    # mutual e1_neighbors
                    csls_links.append([true_e0, e1])

        if len(csls_links) == 0:
            return None
        else:
            csls_links = torch.LongTensor(csls_links)

            # propagate to both subgraph_list_KG and subgraph_list_align    # TODO： whether need to sample and propagate to align_list？
            subgrarph_list_from_alignment(csls_links, kg0, kg1, is_kg_list=True)
            subgrarph_list_from_alignment(csls_links, kg0, kg1, is_kg_list=False)

            return csls_links





