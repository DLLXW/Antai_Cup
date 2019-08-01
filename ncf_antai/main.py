# coding=utf-8
import sys
import pickle
sys.path.append("../../")
import time
import os
import shutil
#import papermill as pm
import pandas as pd
import numpy as np
import tensorflow as tf
from ncf_singlenode import NCF
from dataset import Dataset as NCFDataset
from python_splitters import python_chrono_split
from python_evaluation import (rmse, mae, rsquared, exp_var, map_at_k, ndcg_at_k, precision_at_k, 
                                                    recall_at_k, get_top_k_items)
import random
import numpy as np
import pandas as pd
import warnings
from time import time
import logging
from constants import (
    DEFAULT_ITEM_COL,
    DEFAULT_USER_COL,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
)
                                                     

print("System version: {}".format(sys.version))
print("Pandas version: {}".format(pd.__version__))
print("Tensorflow version: {}".format(tf.__version__))

# top k items to recommend
TOP_K = 10

# Model parameters
BATCH_SIZE_DATA = 10000#每批次读入数据块的大小
BATCH_SIZE_TRAIN = 1024#训练时的batch_size
ROUND=101  #每一批数据迭代的轮数,影响训练速度和loss下降速度
n_users=61211 
n_items=28348
model_type="neumf",
n_factors=4,
layer_sizes=[16,8,4],
n_epochs=100,
learning_rate=1e-3,
verbose=10,
seed=42
#model_para=[n_users,n_items,n_factors,]
"""Constructor

Args:
    n_users (int): Number of users in the dataset.
    n_items (int): Number of items in the dataset.
    model_type (str): Model type.
    n_factors (int): Dimension of latent space.
    layer_sizes (list): Number of layers for MLP.
    n_epochs (int): Number of epochs for training.
    batch_size (int): Batch size.
    learning_rate (float): Learning rate.
    verbose (int): Whether to show the training output or not.
    seed (int): Seed.

"""

print('set model parameters finished')

logger = logging.getLogger(__name__)
MODEL_CHECKPOINT = "model.ckpt"
model_save_dir='./model'


# seed
tf.set_random_seed(seed)
np.random.seed(seed)



# check model type
'''
model_options = ["gmf", "mlp", "neumf"]
if model_type not in model_options:
    raise ValueError(
        "Wrong model type, please select one of this list: {}".format(
            model_options
        )
    )
'''
# ncf layer input size
ncf_layer_size =8 #n_factors + layer_sizes[-1]
# set GPU use with demand growth
gpu_options = tf.GPUOptions(allow_growth=True)


#create model
def model(user_input,item_input,labels,n_users,n_items,n_factors,seed,learning_rate,layer_sizes):
    # reset graph
    #tf.reset_default_graph()

    with tf.variable_scope("embedding"):

        # set embedding table
        embedding_gmf_P = tf.Variable(
            tf.truncated_normal(
                shape=[n_users, n_factors], mean=0.0, stddev=0.01, seed=seed,
            ),
            name="embedding_gmf_P",
            dtype=tf.float32,
        )

        embedding_gmf_Q = tf.Variable(
            tf.truncated_normal(
                shape=[n_items, n_factors], mean=0.0, stddev=0.01, seed=seed,
            ),
            name="embedding_gmf_Q",
            dtype=tf.float32,
        )

        # set embedding table
        embedding_mlp_P = tf.Variable(
            tf.truncated_normal(
                shape=[n_users, int(layer_sizes[0] / 2)],
                mean=0.0,
                stddev=0.01,
                seed=seed,
            ),
            name="embedding_mlp_P",
            dtype=tf.float32,
        )

        embedding_mlp_Q = tf.Variable(
            tf.truncated_normal(
                shape=[n_items, int(layer_sizes[0] / 2)],
                mean=0.0,
                stddev=0.01,
                seed=seed,
            ),
            name="embedding_mlp_Q",
            dtype=tf.float32,
        )

    with tf.variable_scope("gmf"):

        # get user embedding p and item embedding q
        gmf_p = tf.reduce_sum(
            tf.nn.embedding_lookup(embedding_gmf_P, user_input), 1
        )
        gmf_q = tf.reduce_sum(
            tf.nn.embedding_lookup(embedding_gmf_Q, item_input), 1
        )

        # get gmf vector
        gmf_vector = gmf_p * gmf_q

    with tf.variable_scope("mlp"):

        # get user embedding p and item embedding q
        mlp_p = tf.reduce_sum(
            tf.nn.embedding_lookup(embedding_mlp_P, user_input), 1
        )
        mlp_q = tf.reduce_sum(
            tf.nn.embedding_lookup(embedding_mlp_Q, item_input), 1
        )

        # concatenate user and item vector
        output = tf.concat([mlp_p, mlp_q], 1)

        # MLP Layers
        for layer_size in layer_sizes[1:]:
            output = tf.contrib.layers.fully_connected(
                output,
                num_outputs=layer_size,
                activation_fn=tf.nn.relu,
                weights_initializer=tf.contrib.layers.xavier_initializer(seed=seed),
            )
        mlp_vector = output

        # self.output = tf.sigmoid(tf.reduce_sum(self.mlp_vector, axis=1, keepdims=True))

    with tf.variable_scope("ncf"):
            
        # concatenate GMF and MLP vector
        ncf_vector = tf.concat([gmf_vector, mlp_vector], 1)
        # get predicted rating score
        output = tf.contrib.layers.fully_connected(
            ncf_vector,
            num_outputs=1,
            activation_fn=None,
            biases_initializer=None,
            weights_initializer=tf.contrib.layers.xavier_initializer(seed=seed),
        )
        output = tf.sigmoid(output)

    with tf.variable_scope("loss"):

        # set loss function
        loss = tf.losses.log_loss(labels, output)

    with tf.variable_scope("optimizer"):

        # set optimizer
        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate
        ).minimize(loss)
    return output,loss,optimizer


  
def main():
    user_input = tf.placeholder(tf.int32, shape=[None, 1])
    item_input = tf.placeholder(tf.int32, shape=[None, 1])
    labels = tf.placeholder(tf.float32, shape=[None, 1])
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        output,loss,optimizer=model(user_input,item_input,labels,61211,28348,4,42,0.001,[16,8,4])
        sess.run(tf.global_variables_initializer()) # parameters initialization
        data_batchs= pd.read_csv('./train_ncf.csv', chunksize=BATCH_SIZE_DATA)
        print('read data finished...')
        
        count = 0
        m=0
        train_begin = time()
        #train_loss = []
        for data_batch in data_batchs:
            m=m+1
            
            #split train/test for this data_batch
            train, test = python_chrono_split(data_batch, 0.75)
            data = NCFDataset(train=train, test=test, seed=42)
            print('第%d批数据处理完毕......'% m)
            # get user and item mapping dict
            user2id = data.user2id
            item2id = data.item2id
            id2user = data.id2user
            id2item = data.id2item
            

            # loop for n_epochs
            for epoch_count in range(1,ROUND): #把每批次数据训练ROUND-1次
                train_loss = []
                # negative sampling for training                
                data.negative_sampling()                
                # calculate loss and update NCF parameters
                for usr, itm, rating in data.train_loader(BATCH_SIZE_TRAIN):
                    usr = np.array([user2id[x] for x in usr])
                    itm = np.array([item2id[x] for x in itm])
                    rating= np.array(rating)
                    feed_dict = {
                        user_input: usr[..., None],
                        item_input: itm[..., None],
                        labels: rating[..., None],
                    }

                    # get loss and execute optimization
                    losses, _ = sess.run([loss, optimizer], feed_dict)
                    train_loss.append(losses)
                     
            print('loss:%f' %np.mean(train_loss))#打印的是第m批数据的最后一轮迭代的1024个样本的平均训练loss
        train_time = time() - train_begin
        print('.........................')
        print('Time Consumption:%f seconds'  % train_time )
        print('.........................')
        saver = tf.train.Saver()
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        saver.save(sess, os.path.join(model_save_dir, MODEL_CHECKPOINT))
        print('Save model finished......')
        print('.........................')
        #prediction
        print('Prediction and Evaluation begining......')
        k = TOP_K

        ndcgs = []
        hit_ratio = []

        
        for pre_usr, pre_itm,Pre_rating in data.test_loader():
            pre_user_input = np.array([user2id[x] for x in pre_usr])
            pre_item_input = np.array([item2id[x] for x in pre_itm])
            #Pre_rating= np.array(Pre_rating)
            # get feed dict
            feed_dict = {
                user_input: pre_user_input[..., None],
                item_input: pre_item_input[..., None],
                #labels: Pre_rating[..., None],
                }
            pre_output=sess.run(output,feed_dict)
            pre_output=pre_output.reshape(-1)
            pre_output = np.squeeze(pre_output)
            rank = sum(pre_output >= pre_output[0])
            if rank <= k:
                ndcgs.append(1 / np.log(rank + 1))
                hit_ratio.append(1)
            else:
                ndcgs.append(0)
                hit_ratio.append(0)

        eval_ndcg = np.mean(ndcgs)
        eval_hr = np.mean(hit_ratio)

        print("HR:\t%f" % eval_hr)
        print("NDCG:\t%f" % eval_ndcg)
        print('Prediction and Evaluation end......')
      
        
            
             

if __name__ == '__main__':
    main()

