from __future__ import division
import time
import pandas as pd
import numpy as np

import logging
from os.path import join
import torch
from src.utils import nodes_to_graph,Ranking_all_batch



class Tester:
    def __init__(self, target_kg, supporter_kgs,model,device,data_dir):
        """
        :param target_kg: KnowledgeGraph object
        :param support_kgs: list[KnowledgeGraph]
        """
        self.target_kg = target_kg
        self.supporter_kgs = supporter_kgs
        self.device = device
        self.model = model
        self.data_dir = data_dir

       
    def get_hit_mrr(self,topk_indices_all,ground_truth):

        # ground_truth = ground_truth.repeat(1,kg.num_entity) #[n_test,n]
        zero_tensor = torch.tensor([0]).to(ground_truth.device)
        one_tensor = torch.tensor([1]).to(ground_truth.device)

        # Calculate Hit@1, Hit@10
        hits_1 = torch.where(ground_truth == topk_indices_all[:, [0]], one_tensor, zero_tensor).sum().item()
        hits_10 = torch.where(ground_truth == topk_indices_all[:, :10], one_tensor, zero_tensor).sum().item()


        # Calculate MRR
        gt_expanded = ground_truth.expand_as(topk_indices_all)
        hits = (gt_expanded == topk_indices_all).nonzero()
        ranks = hits[:, -1] + 1
        ranks = ranks.float()
        rranks = torch.reciprocal(ranks)
        mrr = torch.sum(rranks).data / ground_truth.size(0)

        return hits_1,hits_10,mrr
    
    def test(self,args,is_val=True,is_lifted = False):
        """
        # for validation set!!

        Compute Hits@10 on first param.n_test test samples
        :param supporter_kg: needed when mode == KG1 or LINK_REDIRECT. None for KG0 or VOTING
        :param voting_function: used when mode==VOTING. Default: vote by count
        :return:
        """

        time0 = time.time()
        if is_val:
            samples = self.target_kg.h_val.shape[0]
            ground_truth = self.target_kg.t_val.view(-1,1).to(self.device)
            output_text = "Val:"
            kg_batch_generator = self.target_kg.generate_batch_data(self.target_kg.h_val, self.target_kg.r_val, self.target_kg.t_val, batch_size=args.batch_size, shuffle=False)
            
        else:
            samples = self.target_kg.h_test.shape[0]
            ground_truth = self.target_kg.t_test.view(-1,1).to(self.device)
            output_text = "Test:"
            kg_batch_generator = self.target_kg.generate_batch_data(self.target_kg.h_test, self.target_kg.r_test, self.target_kg.t_test, batch_size=args.batch_size, shuffle=False)
            
            
        total_entity_num = self.target_kg.num_entity
        self.pre_compute_all_embeddings(args.batch_size) # compute embeddings
                
        topk_indices_all = []

        for kg_batch_each in kg_batch_generator:
            h_batch = kg_batch_each[:, 0].view(-1)
            r_batch = kg_batch_each[:, 1].to(self.device)  # global index
            h_embedding = self.target_kg.computed_entity_embedidng_KG[h_batch,:]
            model_predictions = self.model.predict(h_embedding, r_batch)
            model_predictions = torch.squeeze(model_predictions,dim=1)
            ranking_indices, ranking_scores = Ranking_all_batch(model_predictions, self.target_kg.computed_entity_embedidng_KG) 
        
            topk_indices_all.append(ranking_indices)

        pred_all = torch.cat(topk_indices_all,dim=0)
        assert pred_all.shape[1] == total_entity_num
        assert pred_all.shape[0] == samples
        
         # Lifted Setting
        if is_lifted and not is_val: # test for lifted setting
            hr2t_train = hr2t_from_train_set(self.data_dir + 'kg', self.target_kg.lang)
            testfile = join(self.data_dir + '/kg', self.target_kg.lang + '-test.tsv')
            testcases = pd.read_csv(testfile, sep='\t', header=None).values.astype(np.int)
            

            for i,each_testcase in enumerate(testcases):
                
                h_each = int(each_testcase[0])
                r_each = int(each_testcase[1])
                if (h_each, r_each) in hr2t_train:
                    tmp = torch.LongTensor([e for e in pred_all[i,:] if int(e.detach().data.item()) not in hr2t_train[(h_each,r_each)]]).to(self.device)# a short list
                    
                    tmp_length = tmp.shape[0]
                    
                    pred_all[i,:] = -1*torch.ones_like(pred_all[i,:])
                    pred_all[i,:tmp_length] = tmp.to(self.device)
                    

        hits_1_compute,hits_10_compute, mrr = self.get_hit_mrr(pred_all,ground_truth)
        
        hits_1_ratio = hits_1_compute/samples
        hits_10_ratio = hits_10_compute/samples
        
        # logging.info('===Validation %s===' % mode)
        
        if is_lifted and not is_val:
            logging.info('%s Hits@%d (%d triples,lifted): %f' % (output_text, 1, samples, hits_1_ratio))
            logging.info('%s Hits@%d (%d triples,lifted): %f' % (output_text, 10, samples, hits_10_ratio))
            logging.info('%s MRR (%d triples,lifted): %f' % (output_text,samples, mrr))
        else:
            logging.info('%s Hits@%d (%d triples): %f' % (output_text, 1, samples, hits_1_ratio))
            logging.info('%s Hits@%d (%d triples): %f' % (output_text, 10, samples, hits_10_ratio))
            logging.info('%s MRR (%d triples): %f' % (output_text,samples, mrr))
        print('time: %s' % (time.time() - time0))
        
        return [hits_1_ratio,hits_10_ratio,mrr]

    # def get_kg_embeddings_matrix(self,kg,batch_size,device):
    #     # All nodes in the dataset
    #     # model can be in cuda and outside of cuda
    #     node_index_tensor = torch.LongTensor([i for i in range(kg.num_entity)])
    #     graphs = nodes_to_graph(kg.subgraph_list_kg,node_index_tensor,batch_size)
    #
    #     embedding_list = []
    #     for graph_batch in graphs:
    #         assert graph_batch.edge_index.shape[1] == graph_batch.edge_attr.shape[0]
    #         graph_batch = graph_batch.to(device) # only used to retrive relations
    #         node_embeddings = self.model.forward_GNN_embedding(graph_batch,self.model.encoder_KG)
    #         embedding_list.append(node_embeddings)
    #
    #     embedding_table = torch.cat(embedding_list,dim=0).to(device) #[n,d]
    #
    #     return embedding_table
       
    def pre_compute_all_embeddings(self,batch_size):
        # no gradient compute!
        with torch.no_grad():
            self.target_kg.computed_entity_embedidng_KG = self.model.get_kg_embeddings_matrix(self.target_kg,batch_size,self.device,is_kg= True)
        
#             for supporter_KG in self.supporter_kgs:
#                 self.supporter_KG.computed_entity_embedidng_KG = self.get_kg_embeddings_matrix(self.target_kg,batch_size,self.device,is_aligned = False)


def extract_entities(id_score_tuples):
    return [ent_id for ent_id, score in id_score_tuples]


def filt_hits_at_n(results, lang, hr2t_train, n):
    """
    Filtered setting Hits@n when testing
    :param hr2t_train: {(h,r):set(t)}
    :param results: df, h,r,t, lang
    :return:
    """
    hits = 0
    for index, row in results.iterrows():
        t = row['t']
        predictions = row[lang]  # list[(entity,socre)]

        predictions = extract_entities(predictions)
        if (row['h'], row['r']) in hr2t_train:  # filter
            h, r = row['h'], row['r']
            predictions = [e for e in predictions if e not in hr2t_train[(h,r)]]
        predictions = predictions[:n]  # top n
        if t in predictions:
            hits += 1
    hits_ratio = hits / results.shape[0]
    logging.info('Hits@%d (%d triples)(filt): %.4f' % (n, results.shape[0], hits_ratio))
    return hits_ratio




def hr2t_from_train_set(data_dir, target_lang):
    train_df = pd.read_csv(join(data_dir, f'{target_lang}-train.tsv'), sep='\t')
    tripleset = set([tuple([h,r,t]) for h,r,t in (train_df.values)])

    hr2t = {}  # {(h,r):set(t)}
    for tp in tripleset:
        h,r,t=int(tp[0]),int(tp[1]),int(tp[2])
        if (h,r) not in hr2t:
            hr2t[(h,r)] = set()
        hr2t[(h,r)].add(t)
    return hr2t




