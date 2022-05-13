# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python
# coding: utf-8

"""
Working directory: project root

In all variable names, 0 denotes the target/sparse kg and 1 denotes the source/dense kg.
"""



import os
from os.path import join

print('Current working dir', os.getcwd())
import sys

if './' not in sys.path:
    sys.path.append('./')

import torch
import src.param as param
from src.data_loader import ParseData
import logging
import argparse
import torch.optim as optim
from src.sg_kge_model import SG_KGE
import numpy as np
from torch.utils.data import DataLoader
from src.utils import save_model,load_model,update_edge_index_type,test_kg,nodes_to_graph,get_kg_embeddings_matrix,get_negative_samples_simple
import src.param as param



def set_logger(model_dir,alias):
    '''
    Write logs to checkpoint and console
    '''

    log_file = model_dir + 'alias.log'

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)



def set_args(args):

    if args.use_default:
        if args.knowledge_model == 'transe':
#             # For debug usage
#             args.epoch10 = 1
#             args.epoch11 = 1
#             args.epoch2 = 1
#             args.lr = 1e-3
#             args.round = 1

            if args.initialization_method != "bert_initialized":
                args.dim = 100
            
            args.epoch10 = 10
            args.epoch11 = 10
            args.epoch2 = 5
            args.lr = 1e-3
            args.encoder_hdim_kg = 128
            args.encoder_hdim_align = 128
            
            if args.initialization_method != "bert_initialized":
                args.dim = 100
            args.round = 2
            
        elif args.knowledge_model == 'rotate':
            param.epoch10 = 100
            param.epoch11 = 100
            param.epoch2 = 5
            param.lr = 1e-2
            param.dim = 400
            param.round = 3

            args.lr = param.lr
            args.dim = param.dim

            # # For debug usage
            # param.epoch10 = 1
            # param.epoch11 = 1
            # param.epoch2 = 1
            # param.lr = 1e-2
            # param.dim = 400
            # param.round = 1
    

    return args

def train_align_batch(model,args,lang0,lang1,seed_links_each,optimizer,sub_graph_list,epoch,node_id_base):
    # TODO: can seperate the batch size w.r.t KGE loss
    align_data_loader = DataLoader(seed_links_each,batch_size=args.batch_size, shuffle=True)
    for one_epoch in range(epoch):
        align_loss = []
        for align_each in align_data_loader:
            optimizer.zero_grad()
            graph_e0 = nodes_to_graph(node_id_base[lang0],sub_graph_list,align_each[:, 0],batch_size = -1).to(args.device)
            graph_e1 = nodes_to_graph(node_id_base[lang1],sub_graph_list,align_each[:, 1],batch_size = -1).to(args.device)
        
            loss = model.forward_align(graph_e0, graph_e1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            align_loss.append(loss.item())

            del loss
            torch.cuda.empty_cache()

        logging.info('Align {:s} {:s} Epoch {:d} [Train Align Loss {:.6f}|'.format(
            lang0,
            lang1,
            one_epoch,
            np.mean(align_loss)))
            

def train_eachkg_batch(model,kg,optimizer,args,node_id_base,sub_graph_list,epoch):
    kg_batch_generator = kg.generate_batch_data(kg.h_train,kg.r_train,kg.t_train,batch_size = args.batch_size,shuffle=False)
    lang = kg.lang
    
    for one_epoch in range(epoch):
        kg_loss = []
        for kg_batch_each in kg_batch_generator:
            # get negative sample
            kg_batch_each = get_negative_samples_simple(kg_batch_each,kg.num_entity)
            # generate edge_index_each, edge_value_each, x to serve as the GNN input.
            graph_h = nodes_to_graph(node_id_base[lang],sub_graph_list,kg_batch_each[:,0],batch_size = -1).to(args.device)
            graph_t = nodes_to_graph(node_id_base[lang],sub_graph_list,kg_batch_each[:,2],batch_size = -1).to(args.device)
            graph_t_neg = nodes_to_graph(node_id_base[lang],sub_graph_list,kg_batch_each[:,3],batch_size = -1).to(args.device)
            kg_batch_each = kg_batch_each.to(args.device) # only used to retrive relations
     
            
            optimizer.zero_grad()
            loss = model.forward_KG(graph_h,kg_batch_each,graph_t,graph_t_neg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            kg_loss.append(loss.item())

            del loss
            torch.cuda.empty_cache()

        logging.info('KG {:s} Epoch {:d} [Train KG Loss {:.6f}|'.format(
            kg.lang,
            one_epoch,
            np.mean(kg_loss)))



        
        

parser = argparse.ArgumentParser(
    description='Training and Testing Knowledge Graph Embedding Models',
    usage='run.py [<args>] [-h | --help]'
)
# Task
parser.add_argument('-l', '--target_language', type=str, default = 'el', choices=['ja', 'el', 'es', 'fr', 'en'], help="target kg")

# Hyper parameters
parser.add_argument('-m', '--knowledge_model', default='transe', type=str, choices=['transe', 'rotate'])
parser.add_argument('-d', '--dim', default=100, type=int,help='dimension of the entity embeddings')
parser.add_argument('--encoder_hdim_kg', default=128, type=int, help='dimension of GNN for KGC')
parser.add_argument('--encoder_hdim_align', default=128, type=int, help='dimension of GNN for align loss')
parser.add_argument('--transe_margin', default=0.3, type=float)
parser.add_argument('--rotate_gamma', default=24, type=float)
parser.add_argument('--entity_emb', default='title', type=str,  choices=['title', 'descrip'], help="embedding text for entities")
parser.add_argument('--preserved_percentage', default=0.9, type=float, help="how many align links to preserve when generating GNN initial input graph")
parser.add_argument('--num_hop', default=2, type=int,
                    help="hop sampling")
parser.add_argument('--initialization_method', default="random", type=str, choices=['bert_initialized', 'bert_w','random'],
                    help="how to initialize entity and relation embeddings")
parser.add_argument('--k_csls', default=3, type=int,
                    help="how many nearest neighbors to compute csls")
parser.add_argument('--n_heads', default=1, type=int,
                    help="heads in each GNN layer")
parser.add_argument('--n_layers_KG', default=1, type=int,
                    help="GNN layer for KGE")
parser.add_argument('--n_layers_align', default=1, type=int,
                    help="GNN layer for KGE")

parser.add_argument('--load', default=None, type=str, help="load exsiting models")


parser.add_argument('--use_default', action="store_true", help="Use default setting. This will override every setting except for targe_langauge and knowledge_model")

# Optimization
parser.add_argument('--lr', default=1e-2, type=float, help="learning rate for knowledge model")
parser.add_argument('--align_lr', default=1e-3, type=float, help="learning rate for knowledge model")
parser.add_argument('-b', '--batch_size', default=512, type=int, help="batch size of queries")
parser.add_argument('--optimizer', type=str, default="Adam", help='Adam, AdamW')
parser.add_argument('--reg_scale', default=1e-5, type=float, help="scale for regularization")
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
parser.add_argument('--val_freq', type=int, default=1, help='Dropout rate (1 - keep probability).')
parser.add_argument('--clip', type=float, default=10, help='Gradient Norm Clipping')



# Steps
parser.add_argument('--epoch2', default=5, type=int, help="how many align model epoch to train before switching to knowledge model")
parser.add_argument('--epoch11', default=10, type=int, help="how many knowledge model epochs for each align epoch")
parser.add_argument('--epoch10', default=10, type=int, help="how many knowledge model epochs for each align epoch")
parser.add_argument('--round', default=5, type=int,
                    help="how many rounds to train")

parser.add_argument('--alias', default="run", type=str,
                    help="how many rounds to train")

parser.add_argument('--data_path', default="data/", type=str,
                    help="how many rounds to train")

      
        
        
        
if __name__ == '__main__':
    args = parser.parse_args()  
    print(args)
    
     ########### CPU AND GPU related, Mode related, Dataset Related
    if torch.cuda.is_available():
        print("Using GPU" + "-" * 80)
        args.device = torch.device("cuda:0")
    else:
        print("Using CPU" + "-" * 80)
        args.device = torch.device("cpu")



    # Dim for relations and entities
    args.entity_dim = args.dim
    if args.knowledge_model == "rotate":
        args.relation_dim = args.dim//2

    else:
        args.relation_dim = args.entity_dim
                      
      # if using bert to initialize, cannot use rotate, dim = 768
    if args.initialization_method== "bert_initialized":
        args.entity_dim = 768
        args.relation_dim = 768
        assert args.knowledge_model != "rotate"

                      

    # LOAD DATA
    dataloader = ParseData(data_path = args.data_path, target_kg = args.target_language, is_unified = False)
    edge_index,edge_type, entity_bert_emb, relation_bert_emb, seeds_masked, seeds_all, seeds_preserved, kg_object_dict, kg_node_base_dict, kg_entity_num_dict,num_relations,sub_graph_list = dataloader.load_data(args,num_hop = args.num_hop,k=10)
    

    
    args.num_relations = num_relations
    target_lang = args.target_language
    
   
    model_dir = join('./trained_model_no_generation', f'kens-{args.knowledge_model}-{args.dim}-{args.initialization_method}', target_lang)  # output
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # logging
    set_logger(model_dir,args.alias)  
    logging.info(args)
    logging.info('Knowledge model: %s'%(args.knowledge_model))
    logging.info('target language: %s'%(args.target_language))
    logging.info('initialization method: %s' % (args.initialization_method))

    # hyper-parameters
    logging.info(f'lr: {args.lr}')
    logging.info('entity dim: %s' % (args.entity_dim))
    logging.info('GNN dim: %s' % (args.encoder_hdim_kg))
    logging.info('k_csls: %s' % (args.k_csls))

    # Model
    model = SG_KGE(args,entity_bert_emb,relation_bert_emb,kg_node_base_dict,kg_entity_num_dict,args.device)
    model = model.to(args.device)
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    #Load model
    if args.load is not None:
        ckpt_path = os.path.join(args.save, args.load)
        load_model(model, ckpt_path, args.device)

    # Optimizer for all KGs
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print('model initialization done')


    ############ Start Training !!!
    best_mrr_test = -1
    best_mrr_val = -1
    for i in range(args.round):
        logging.info(f'Epoch: {i}')
        model.train()

        # for kg in all_kgs:
        #     kg.train()
        #     get_entity_embedding(kg) #initial embedding are correct



        # train alignment model
        # Adjust optimizer learning rate
        for param_group in optimizer.param_groups:  # 在每次更新参数前迭代更改学习率
            param_group["lr"] = args.align_lr

        for (lang0,lang1), seed_links_each_masked in seeds_masked.items():
            train_align_batch(model,args,lang0,lang1,seed_links_each_masked,optimizer,sub_graph_list,args.epoch2,kg_node_base_dict)


#         # self-learning: add more pairs to seeds_preserved
#         for (lang0,lang1), seed_links_each in seeds_preserved.items():
#             logging.info(f'self learning[{lang0}][{lang1}]')
#             logging.info("Original link number is %d" % (len(seed_links_each)))
#             num_nodes0 = kg_entity_num_dict[lang0]
#             num_nodes1 = kg_entity_num_dict[lang1]
#             emb0 = get_kg_embeddings_matrix(model,lang0,num_nodes0,kg_node_base_dict,args.batch_size,sub_graph_list,args.device,is_aligned = True)
#             emb1 = get_kg_embeddings_matrix(model,lang1,num_nodes1,kg_node_base_dict,args.batch_size,sub_graph_list,args.device,is_aligned = True)
            
#             found,found_renumbered = model.extend_seed_align_links(lang0,lang1,emb0,emb1,seed_links_each,args.device,args.k_csls)
            
#             if len(found) > 0:  # not []
#                 new_seeds = torch.cat([seed_links_each, found], axis=0)
#                 seeds_preserved[(lang0, lang1)] = new_seeds
#                 logging.info("Generated link number is %d" % (len(found)))

#                 # Update edge_index, edge_type TODO!
#                 edge_index_new, edge_type_new = update_edge_index_type(found_renumbered,num_relations)
#                 edge_index = np.concatenate([edge_index,edge_index_new],axis=1)
#                 edge_type = np.concatenate([edge_type,edge_type_new])

#             del em0,emb1
#             torch.cuda.empty_cache()

        # # train knowledge model
        # Adjust learning rate for kg module
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr
        
#         # Train the targeted KG
        kg_target = kg_object_dict[target_lang]
        train_eachkg_batch(model,kg_target,optimizer,args,kg_node_base_dict,sub_graph_list,args.epoch10)
        #Train supporter KG
        for lang,kg in kg_object_dict.items():
            if lang != target_lang:
                train_eachkg_batch(model,kg,optimizer,args,kg_node_base_dict,sub_graph_list,args.epoch11)



        if i % args.val_freq == 0:  # validation
            logging.info(f'=== round {i}')
            logging.info(f'[{target_lang}]')

            model.eval()
            mrr_trains = test_kg(model,"data/kg/",args.batch_size, sub_graph_list,args.device, kg_object_dict[target_lang], is_train=True)
            mrr_val = test_kg(model,"data/kg/",args.batch_size, sub_graph_list,args.device, kg_object_dict[target_lang], is_val=True)
            mrr_test = test_kg(model,"data/kg/",args.batch_size, sub_graph_list, args.device,kg_object_dict[target_lang], is_val=False)
            
             # also test other kgs
            for kg_name in kg_node_base_dict:
                if kg_name!= target_lang:
                    mrr_test = test_kg(model,"data/kg/",args.batch_size, sub_graph_list, args.device,kg_object_dict[kg_name], is_val=False)
           

            if best_mrr_test< mrr_test:
                best_mrr_test = mrr_test
                best_mrr_val = mrr_val
                logging.info("Best MRR on Test!")
                save_model(model,model_dir, args.target_language + args.alias + "_mrr_test_" + str(best_mrr_test) + ".ckpt")

    #TODO: write down the results


