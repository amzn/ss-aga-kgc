# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import logging
import torch
from src.sg_kge_model import Ranking_all_batch
import  numpy as np
import pandas as pd
from os.path import join
from torch_geometric.data import DataLoader as Loader



def get_negative_samples_simple(kg_batch_each,num_entity):
    batch_size = kg_batch_each.shape[0]
    rand_negs = torch.randint(high=num_entity, size=(batch_size,))
    
    rand_negs = rand_negs.view(-1)
    kg_batch_each[:,3] = rand_negs
    return kg_batch_each


def save_model(model, output_dir,output_name):

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	# save the weights for the whole model
	ckpt_path = os.path.join(output_dir, output_name)
	torch.save({
		'state_dict': model.state_dict(),
	}, ckpt_path)


def load_model(model, ckpt_path,device):
	"""
    Recreate the knowledge model and load the weights
    :param path:
    :return: TODO: here we only load the parameterfor the kg model, but later on should load bert and GNN as well.
    """
	if not os.path.exists(ckpt_path):
		raise Exception("Checkpoint " + ckpt_path + " does not exist.")
	checkpt = torch.load(ckpt_path)
	state_dict = checkpt['state_dict']
	model_dict = model.state_dict()

	# 1. filter out unnecessary keys
	state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
	# 2. overwrite entries in the existing state dict
	model_dict.update(state_dict)
	# 3. load the new state dict
	model.load_state_dict(state_dict)
	model.to(device)

def test_kg(model,kg_data_dir,batch_size, sub_graph_list, device, kg, is_val=True,is_train = False):
	"""

    Compute Hits@10 on first param.n_test test samples
    :param mode: TestMode. For LINK_TRANSFER, triples without links will be skipped
    :param supporter_kg: needed when mode == KG1 or LINK_REDIRECT. None for KG0 or VOTING
    :param voting_function: used when mode==VOTING. Default: vote by count
    :return:
    """
    
	if is_train:
		kg_batch_generator = kg.generate_batch_data_test_val(kg.h_train, kg.r_train, kg.t_train, batch_size=batch_size,
													shuffle=False)
	elif is_val:
		kg_batch_generator = kg.generate_batch_data_test_val(kg.h_val, kg.r_val, kg.t_val, batch_size=batch_size,
													shuffle=False)
	else:
		kg_batch_generator = kg.generate_batch_data_test_val(kg.h_test, kg.r_test, kg.t_test, batch_size=batch_size,
													shuffle=False)

	with torch.no_grad():
        
		model.eval()
		entity_embeddings =get_kg_embeddings_matrix(model,kg.lang,kg.num_entity,model.kg_node_base_dict,batch_size,sub_graph_list,device,is_aligned = False)

		topk_indices_all = []
		topk_scores_all = []
		lang_base = model.kg_node_base_dict[kg.lang]

		for kg_batch_each in kg_batch_generator:
			
			h_batch = kg_batch_each[:, 0]
			r_batch = kg_batch_each[:, 1].to(device)
			h_graph_input = nodes_to_graph(lang_base,sub_graph_list,h_batch).to(device)

			model_predictions = model.predict(h_graph_input, r_batch)

			ranking_indices, ranking_scores = Ranking_all_batch(model_predictions, entity_embeddings)  # [b,N]
        
			topk_indices_all.append(ranking_indices)
			topk_scores_all.append(ranking_scores)

        
		topk_indices_all = torch.cat(topk_indices_all, dim=0)  # [n_test,n]
		topk_scores_all = torch.cat(topk_scores_all, dim=0)  # [n_test,n]
    
        
		if is_train:
			ground_truth = kg.t_train.view(-1, 1).to(device)  # [n_test,1]
		elif is_val:
			ground_truth = kg.t_val.view(-1, 1).to(device)  # [n_test,1]
		else:
			ground_truth = kg.t_test.view(-1, 1).to(device)  # [n_test,1]

		assert topk_scores_all.shape[1] == kg.num_entity

		hits_1, hits_10, mrr = get_hit_mrr(topk_indices_all,ground_truth,device)
		hits_1_filtered, hits_10_filtered, mrr_filtered = get_hit_mrr(topk_indices_all, ground_truth, device)
		total_sample_num = topk_indices_all.shape[0]

		logging.info('%s: Hits@%d (%d triples): %f' % (kg.lang, 1, total_sample_num, hits_1 / total_sample_num))
		logging.info('%s: Hits@%d (%d triples): %f' % (kg.lang, 10, total_sample_num, hits_10 / total_sample_num))
		logging.info('%s: MRR (%d triples): %f' % (kg.lang, total_sample_num, mrr))


	return mrr

def update_edge_index_type(found_renumbered,num_relations):
	# Update edge_index, edge_type
	sender_list = []
	receiver_list = []
	weight_list = []

	for each_found in found_renumbered:
		align_id0 = each_found[0]
		align_id1 = each_found[1]

		sender_list.append(int(align_id0))
		receiver_list.append(align_id1)
		weight_list.append(num_relations)

		sender_list.append(int(align_id1))
		receiver_list.append(align_id0)
		weight_list.append(num_relations)

	edge_index_new = np.vstack((sender_list, receiver_list))
	edge_type_new = np.asarray(weight_list)

	return edge_index_new,edge_type_new



def hr2t_from_train_set(data_dir, target_lang):
    train_df = pd.read_csv(join(data_dir, f'{target_lang}-train.tsv'), sep='\t')
    tripleset = set([tuple([h,r,t]) for h,r,t in (train_df.values)])

    hr2t = {}  # {(h,r):set(t)}
    for tp in tripleset:
        h,r,t=tp[0],tp[1],tp[2]
        if (h,r) not in hr2t:
            hr2t[(h,r)] = set()
        hr2t[(h,r)].add(t)
    return hr2t

# def get_lifted_predictions(h,r,hr2t_train,predictions):
# 	pre_filtered = []
#     print(predictions.shape)
# 	for i in range(len(predictions)):
# 		prediction_each = predictions[i]
# 		h_each = int(h[i].item())
# 		r_each = int(r[i].item())
        
# 		if (h_each, r_each) in hr2t_train:  # filter
# 			prediction_each = [e for e in predictions if e not in hr2t_train[(h_each, r_each)]]
# 		pre_filtered.append(prediction_each)

# 	print(type(prediction_each))
# 	print(prediction_each.shape)
# 	return torch.cat(pre_filtered,dim = 0)



def get_hit_mrr(topk_indices_all,ground_truth,device):
	# TODO: check device!
	# ground_truth = ground_truth.repeat(1,kg.num_entity) #[n_test,n]
	zero_tensor = torch.tensor([0], device=device)
	one_tensor = torch.tensor([1], device=device)

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

        



def nodes_to_graph(lang_base,sub_graph_list,node_index,batch_size = -1):
    one_batch = False
    if batch_size == -1:
        # get embeddings together without batch
        batch_size = node_index.shape[0]
        one_batch = True
    
    graphs = [sub_graph_list[i.item() + lang_base] for i in node_index]
    
    graph_loader = Loader(graphs, batch_size=batch_size,shuffle = False)

    if one_batch:
        for one_batch in graph_loader:
            assert one_batch.edge_index.shape[1] == one_batch.edge_attr.shape[0]
            return one_batch
    else:
        return graph_loader
    


def get_kg_embeddings_matrix(model,lang,num_nodes,node_id_base,batch_size,sub_graph_list,device,is_aligned):
    # All nodes in the dataset     
    # model can be in cuda and outside of cuda
    node_index_tensor = torch.LongTensor([i for i in range(num_nodes)])
    graphs = nodes_to_graph(node_id_base[lang],sub_graph_list,node_index_tensor,batch_size)
    
    embedding_list = []
    for graph_batch in graphs:
        assert graph_batch.edge_index.shape[1] == graph_batch.edge_attr.shape[0]
        graph_batch = graph_batch.to(device) # only used to retrive relations  
        if is_aligned:
            node_embeddings = model.forward_GNN_embedding(graph_batch,model.encoder_align)
        else:
            node_embeddings = model.forward_GNN_embedding(graph_batch,model.encoder_KG)
            
        embedding_list.append(node_embeddings)
     
    embedding_table = torch.cat(embedding_list,dim=0).to(device) #[n,d]
    
    return embedding_table
        




