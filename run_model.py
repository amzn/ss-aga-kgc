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
from src.data_loader_new import ParseData
from src.validate import Tester
import numpy as np
import logging
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
from src.utils import nodes_to_graph,nodes_to_graph_align,get_language_list,get_negative_samples_graph,save_model,get_negative_samples_alignment
from src.ssaga_model import SSAGA
from random import SystemRandom
import time



def set_logger(model_dir,args):
    '''
    Write logs to checkpoint and console
    '''
    experimentID = int(SystemRandom().random() * 100000)
    log_file = model_dir+"/train_" + str(experimentID) + "_" + args.alias + ".log"
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

    return experimentID


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='run.py [<args>] [-h | --help]'
    )

    # Data loader related
    parser.add_argument('-l', '--target_language', type=str, default='ja', choices=['ja', 'el', 'es', 'fr', 'en'],
                        help="target kg")
    parser.add_argument('--k', default=10, type=int, help="how many nominations to consider")
    parser.add_argument('--preserved_ratio', default=0.1, type=float, help="how many align links to preserve")
    parser.add_argument('--num_hop', default=2, type=int,
                        help="hop sampling")
    parser.add_argument('--k_csls', default=3, type=int,
                        help="how many nearest neighbors to compute csls")
    parser.add_argument('--data_path', default="dataset", type=str,
                        help="how many rounds to train")
    parser.add_argument('--dataset', default="dbp5l", type=str,
                        help="how many rounds to train")
    parser.add_argument('--load', type=str, default=None, help="name of ckpt. If None, run a new experiment.")
    parser.add_argument('--burin_in_epoch', default=15, type=int, help="how many nominations to consider")
    parser.add_argument('--generation_freq', default=5, type=int, help="how many nominations to consider")

    #KG model related
    parser.add_argument('--transe_margin', default=0.3, type=float)
    parser.add_argument('--align_margin', default=6, type=float)
    parser.add_argument('-d', '--dim', default=256, type=int, help = 'kg embedding table dimension')

    # GNN related
    parser.add_argument('--n_layers_KG', default=2, type=int,help="GNN layer for KGE")
    parser.add_argument('--n_layers_align', default=2, type=int, help="GNN layer for align model")
    parser.add_argument('--encoder_hdim_kg', default=256, type=int, help='dimension of GNN for KGC')
    parser.add_argument('--encoder_hdim_align', default=256, type=int, help='dimension of GNN for align loss')
    parser.add_argument('--n_heads', default=1, type=int, help="GNN layer for KGE")

    # Training Related
    parser.add_argument('--epoch2', default=2, type=int, help="how many align model epoch to train before switching to knowledge model")
    parser.add_argument('--epoch11', default=2, type=int, help="how many knowledge model epochs for each supporter KG")
    parser.add_argument('--epoch10', default=3, type=int, help="how many knowledge model epochs for the target KG")
    parser.add_argument('--round', default=25, type=int,help="how many rounds to train")
    parser.add_argument('--lr', '--learning_rate', default=5e-3, type=float, help="learning rate for knowledge model")
    parser.add_argument('--align_lr', default=1e-3, type=float, help="learning rate for alignment model")
    parser.add_argument('-b', '--batch_size', default=200, type=int, help="batch size of queries")
    parser.add_argument('--optimizer', type=str, default="AdamW", help='Adam, AdamW')
    parser.add_argument('--l2', type=float, default=0, help='l2 regulazer')
    parser.add_argument('--clip', type=float, default=10, help='Gradient Norm Clipping')
    parser.add_argument('--reg_scale', default=1e-5, type=float, help="scale for regularization")
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--val_freq', type=int, default=1, help='Dropout rate (1 - keep probability).')

    # Others
    parser.add_argument('--use_default', action="store_true", help="Use default setting.")
    parser.add_argument('--alias', default="run", type=str, help="how many align links to preserve")

    return parser.parse_args(args)

def set_args(args):

    '''
    Values for the default setting.
    '''
    if args.use_default:
        args.epoch10 = 3
        args.epoch11 = 2
        args.epoch2 = 2
        args.lr = 1e-2
        args.align_lr = 1e-3
        args.encoder_hdim_kg = 256
        args.encoder_hdim_align = 256
        args.dim = 256
        args.round = 25
        args.burin_in_epoch=15
        args.generation_freq=5


    return args



def train_align_batch(args,align_data,optimizer,kg0,kg1,model):

    #TODO: align loss need to have some negative samples as well!

    align_data_loader = DataLoader(align_data,batch_size=args.batch_size, shuffle=True)
    kg_name0 = kg0.lang
    kg_name1 = kg1.lang
    for one_epoch in range(args.epoch2):
        align_loss = []
        for align_each in align_data_loader:
            optimizer.zero_grad()

            e0_graph = nodes_to_graph(kg0.subgraph_list_align,align_each[:,0]).to(args.device)
            e1_graph = nodes_to_graph(kg1.subgraph_list_align,align_each[:,1]).to(args.device)
            batch_size = align_each.shape[0]

           
            e1_neg_index = get_negative_samples_alignment(batch_size,kg1.num_entity) # TODO: negative pair generation
            e1_graph_neg = nodes_to_graph(kg1.subgraph_list_align,e1_neg_index).to(args.device)

            # e1_neg_index = get_negative_samples_alignment(batch_size,kg1.num_entity,args.num_neg_align) # TODO: negative pair generation
            # e1_graph_neg = nodes_to_graph_align(kg1.subgraph_list_align,e1_neg_index).to(args.device)

            loss = model.forward_align(e0_graph, e1_graph,e1_graph_neg)
            loss.backward()
            optimizer.step()

            align_loss.append(loss.item())

            del loss
            torch.cuda.empty_cache()


        logging.info('Align {:s} {:s} Epoch {:d} [Train Align Loss {:.6f}|'.format(
            kg_name0,
            kg_name1,
            one_epoch,
            np.mean(align_loss)))




def train_kg_batch(args,kg,optimizer,num_epoch,model):


    kg_batch_generator = kg.generate_batch_data(kg.h_train,kg.r_train,kg.t_train,batch_size = args.batch_size,shuffle=True)

    for one_epoch in range(num_epoch):
        kg_loss = []
        for kg_batch_each in kg_batch_generator:
            
            h_graph = nodes_to_graph(kg.subgraph_list_kg,kg_batch_each[:,0]).to(args.device)
            t_graph = nodes_to_graph(kg.subgraph_list_kg,kg_batch_each[:,2]).to(args.device)
            batch_size = kg_batch_each.shape[0]
            t_neg_index = get_negative_samples_graph(batch_size,kg.num_entity)
            t_neg_graph = nodes_to_graph(kg.subgraph_list_kg,t_neg_index).to(args.device)
            
            kg_batch_each = kg_batch_each.to(args.device)

            optimizer.zero_grad()
            loss = model.forward_kg(h_graph,kg_batch_each,t_graph,t_neg_graph)
            loss.backward()
            optimizer.step()

            kg_loss.append(loss.item())

            del loss
            torch.cuda.empty_cache()

        logging.info('KG {:s} Epoch {:d} [Train KG Loss {:.6f}|'.format(
            kg.lang,
            one_epoch,
            np.mean(kg_loss)))





def main(args):
    ########### CPU AND GPU related, Mode related, Dataset Related
    if torch.cuda.is_available():
        print("Using GPU" + "-" * 80)
        args.device = torch.device("cuda:0")
    else:
        print("Using CPU" + "-" * 80)
        args.device = torch.device("cpu")

    print(args)
    set_args(args)


    args.entity_dim = args.dim
    args.relation_dim = args.entity_dim



    target_lang = args.target_language
    src_langs = get_language_list(args.data_path + args.dataset)
    src_langs.remove(target_lang)

    # load data

    model_dir = join('./' + args.dataset + "/trained_model", f'SSAGA-{args.dim}', target_lang)  # output TODO: change the name.
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)


    # target (sparse) kg
    dataset = ParseData(args)
    kg_object_dict, seeds_masked, seeds_all, entity_bert_emb = dataset.load_data()
    args.num_relations = dataset.num_relations
    args.num_entities = dataset.num_entities
    args.num_kgs = dataset.num_kgs
    del dataset

    # supporter KG use all links (train and val) to train
    for kg1_name in src_langs:
        kg1 = kg_object_dict[kg1_name]

        kg1.h_train = torch.cat([kg1.h_train, kg1.h_val], axis=0)
        kg1.r_train = torch.cat([kg1.r_train, kg1.r_val], axis=0)
        kg1.t_train = torch.cat([kg1.t_train, kg1.t_val], axis=0)




    # logging
    experimentID = set_logger(model_dir, args)  # set logger
    logging.info('target language: %s' % (target_lang))
    logging.info('set up: %s' % (args.alias))

    logging.info(f'dim: {args.dim}')
    logging.info(f'lr: {args.lr}')
    logging.info(f'preserve_ratio: {args.preserved_ratio}')
    logging.info(f'k: {args.k}')
    logging.info(f'num_hop: {args.num_hop}')
    logging.info(f'k_csls: {args.k_csls}')
    logging.info(f'encoder dim: {args.encoder_hdim_kg}')
    logging.info(f'GNN layer: {args.n_layers_KG}')
    logging.info(f'lr_align: {args.align_lr}')

    logging.info(str(args))


    # Build Model
    model = SSAGA(args, entity_bert_emb, args.num_relations, args.num_entities, args.num_kgs).to(args.device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,weight_decay=args.l2)

    print('model initialization done')



    validator = Tester(kg_object_dict[target_lang], None, model,args.device,args.data_path + args.dataset)
    
    ############ Start Training !!!
    best_test = 0
    best_val = 0
    
    

    for i in range(args.round):
        logging.info(f'Epoch: {i}')

        model.train()
        # Train alignment model (recover masked alignment pairs), Adjust optimizer learning rate
        for param_group in optimizer.param_groups:  # change learning rate
            param_group["lr"] = args.align_lr


        for (kg0_name, kg1_name) in seeds_masked:
            kg0 = kg_object_dict[kg0_name]
            kg1 = kg_object_dict[kg1_name]
            align_links = torch.LongTensor(seeds_masked[(kg0_name, kg1_name)]).to(args.device)
            train_align_batch(args,align_links,optimizer,kg0,kg1,model)


        if (i>=args.burin_in_epoch) and (i%args.generation_freq==0):
            # Self-learning: propose aligned links that have not been seen before.
            model.eval()  # only forward
            with torch.no_grad():
                for (kg0_name, kg1_name) in seeds_all:
                    print(f'self learning[{kg0_name}][{kg1_name}]')
                    time_start = time.time()
                    kg0 = kg_object_dict[kg0_name]
                    kg1 = kg_object_dict[kg1_name]
                    seeds = seeds_all[(kg0_name, kg1_name)]
                    # print("Original link number is %d" % (len(seeds)))
                    found = model.extend_seed_align_links(kg0, kg1, seeds, args.device,
                                                          args.k_csls)  # find new one and update the subgraph_align, subgraph_kg list accordingly.
                    if found != None:
                        new_seeds = torch.cat([seeds, found], axis=0)
                        seeds_all[(kg0_name, kg1_name)] = new_seeds
                        logging.info(
                            "KG {} and KG {} Epoch {:d} Generated link number is {:d}".format(kg0_name, kg1_name, i,
                                                                                              len(found)))

                    logging.info(
                        "KG {} and KG {} Epoch {:d} Generated link number using time {:.2f} secs".format(kg0_name,
                                                                                                         kg1_name, i,
                                                                                                         time.time() - time_start))

            for kg_name_each in kg_object_dict.keys():
                kg_object_dict[kg_name_each].computed_entity_embedidng_align = None



        model.train()
        # Train knowledge model, Adjust learning rate for kg module
        for param_group in optimizer.param_groups:  #
            param_group["lr"] = args.lr

        train_kg_batch(args,kg_object_dict[args.target_language], optimizer, args.epoch10, model)
        for kg1_name in src_langs:
            kg1 = kg_object_dict[kg1_name]
            train_kg_batch(args, kg1, optimizer, args.epoch11, model)



        if i % args.val_freq == 0:  # validation
            logging.info(f'=== round {i}')
            logging.info(f'[{args.target_language}]')

            model.eval()
            with torch.no_grad():
                metrics_val = validator.test(args, is_val=True)  # validation set
                metrics_test = validator.test(args, is_val=False)  # Test set


                if metrics_val[2] > best_val:
                    best_val = metrics_val[2]
                    message_best = 'BestVal! Epoch {:04d} [Test seq] | Best mrr {:.6f}| hits1 {:.6f}| hits10 {:.6f}|'.format(i,metrics_test[2],metrics_test[0],metrics_test[1])
                    logging.info(message_best)
                    # # save model
                    filename = "experiment_" + str(experimentID) + "_preserveRatio_" + str(args.preserved_ratio) + \
                               "_epoch_" + str(i) + "_MRR_" + str(metrics_test[2]) + "_Hit1_" + str(metrics_test[0]) +\
                               "_Hit10_" + str(metrics_test[1]) + '.ckpt'

                    save_model(model, model_dir, filename,args)

                    if metrics_test[2] > best_test: # both best test and best val, save one ckpt, but showing the message
                        message_best = 'BestTest! Epoch {:04d} [Test seq] | Best mrr {:.6f}| hits1 {:.6f}| hits10 {:.6f}|'.format(
                            i, metrics_test[2], metrics_test[0], metrics_test[1])
                        logging.info(message_best)

                elif metrics_test[2] > best_test:
                    best_test = metrics_test[2]
                    message_best = 'BestTest! Epoch {:04d} [Test seq] | Best mrr {:.6f}| hits1 {:.6f}| hits10 {:.6f}|'.format(i,metrics_test[2],metrics_test[0],metrics_test[1])
                    logging.info(message_best)
                    # # save model

                    filename = "experiment_" + str(experimentID) + "_preserveRatio_" + str(args.preserved_ratio) + \
                               "_epoch_" + str(i) + "_MRR_" + str(metrics_test[2]) + "_Hit1_" + str(metrics_test[0]) +\
                               "_Hit10_" + str(metrics_test[1]) + '.ckpt'

                    save_model(model, model_dir, filename,args)














        

if __name__ == "__main__":
    main(parse_args())
    # main(parse_args(['--knowledge_model','transe','--target_language','en','--use_default']))