# Multilingual Knowledge Graph Completion with Self-Supervised  Adaptive Graph Alignment (SS-AGA)

SS-AGA  is a multilingual knowledge graph completion framework that transfers knowledge among multiple KGs sources based on limited seed entity alignment.

You can see our ACL 2022 paper [“**Multilingual Knowledge Graph Completion with Self-Supervised
Adaptive Graph Alignment**”](https://arxiv.org/pdf/2203.14987.pdf) for more details.

This implementation of SS-AGA is based on [Pytorch Geometric](https://github.com/rusty1s/pytorch_geometric) API. Our code is built upon the github [**KEnS**](https://github.com/stasl0217/KEnS) and we thank the authors' effort for making it public.

## Data

**DBP-5L**:  A Public dataset from  https://github.com/stasl0217/KEnS.

**E-PKG**: A new industrial multilingual E-commerce product KG dataset. (To be released.)

Data format: Each dataset contains the following files and folders:

- entity: Folder that contains the entity list for each KG.
- kg: Folder that contains the KG triple list (head_entity_index, relation_index, tail_entity_index) for each kg.
- seed_alignlinks: Folder that contains seed entity alignment pair list between two KGs. 
- relation.txt: File that contains relation that is shared across all KGs.
- entity_embeddings.npy: The numpy file of mbert embedding for each entity from all KGs. Size of [Num_entity_all, 768]. We use the **BERT-Base, Multilingual Cased**  from https://github.com/google-research/bert/blob/master/multilingual.md to generate it.

To run the code, create the folders "dataset/dbp5l",  "dataset/epkg" and download the two datasets respectively.

## Setup

To run the code, you need the following dependencies:

- [Python 3.6.10](https://www.python.org/)

- [Pytorch 1.10.0](https://pytorch.org/)
- [pytorch_geometric 2.0.4](https://pytorch-geometric.readthedocs.io/)
  - torch-cluster==1.6.0
  - torch-scatter==2.0.9
  - torch-sparse==0.6.13
- [numpy 1.16.1](https://numpy.org/)

## Usage

Execute the following scripts to train the model on the targeted japanese KG:

```bash
python run_model.py --target_language ja --use_default
```

There are some key options of this scrips:

- `--target_language`: The targeted KG to conduct the KG completion task.
- `--num_hop`: Number of hops for sampling neighbors for each node.
- `--preserved_ratio` : How many align links to preserve in learning alignment embeddings. The rest are served as masked alignments and we ask the model to recover them.
- `--generation_freq`: How many epochs to conduct new pair generation once.
- `--use_default`:  Use the preset hyper-parameter combinations.

The details of other optional hyperparameters can be found in run_model.py.

## License

Please consider citing the following paper when using our code for your application.

```bibtex
@inproceedings{SS-AGA,
  title={Multilingual Knowledge Graph Completion with Self-Supervised
Adaptive Graph Alignment},
  author={Zijie Huang and Zheng Li and Haoming Jiang and Tianyu Cao and Hanqing Lu and Bing Yin and Karthik Subbian and Yizhou Sun and Wei Wang},
  booktitle={Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2022}
}
```

## License

This project is licensed under the Apache-2.0 License.