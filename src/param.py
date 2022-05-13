# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
epoch10 = 2  # epoch for knowledge model 0
epoch11 = 2  # epoch for knolwedge model 1
epoch2 = 1  # epoch for alignment model
lr = 0.0001
prior_epoch = 100
batch_size = 2048
reg_scale = 1e-5
dim = 400
k = 10
lang = 'ja'
neg_per_pos = 1
n_test = 1000  # quick test for validation
round = 1
align_lr = 1e-3
verbose = False
n_fold = 10
align_ratio = 0.8
val_freq = 1
margin = 0.3

models = ['en', 'ja','fr','es','el']

# model
align_model = 'same'

updating_embedding = True  # True for training, False for testing

json_model = False

knowledge = 'rotate'
# for RotatE
gamma = 24
epsilon = 1e-8
def rotate_embedding_range():
    return (gamma+epsilon)/(dim/2)


# obsolete
src_lang = 'en'
rrf_const = 50
csls = False


