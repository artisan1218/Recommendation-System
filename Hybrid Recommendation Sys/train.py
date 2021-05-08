#!/usr/bin/env python
# coding: utf-8

from pyspark import SparkContext, SparkConf
import random
from itertools import combinations 
import itertools
import math
import sys
import json
import os
import pandas as pd
import numpy as np
import time
from surprise import SVD, Reader, Dataset, dump
import xgboost
import pickle


# In[2]:


startTime = time.time()

os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'

conf = SparkConf().setMaster("local[*]").set("spark.executor.memory", "4g").set("spark.driver.memory", "4g")
sc = SparkContext(conf=conf).getOrCreate()


# In[3]:

train_file_path = "../resource/asnlib/publicdata/train_review.json"
test_groundtruth_path = "./test_review_ratings.json"
friends_path = "../resource/asnlib/publicdata/user.json"
user_avg_file = "../resource/asnlib/publicdata/user_avg.json"
bus_avg_file = "../resource/asnlib/publicdata/business_avg.json"
user_file_path = "../resource/asnlib/publicdata/user.json"

xgb_model_output_path = "./xgb_model.json"
SVD_model_output_path = "./svd_model.model"
item_based_CF_output_path = "./itemBasedCF_itemProfile.json"
content_based_output_path = "./contentBasedModel_itemProfile.json"
uid2idx_output_path = "./uid2idx_dict_model.txt"
bid2idx_output_path = "./bid2idx_dict_model.txt"
friends_output_path = "./friends_model.json"


# ### uid2idx_dict:
# keeps the mapping relation between unique userid to an user index, which is in this format:\
# `
# {
#  '8GWwu6gtDfAFfM2gehfPow': 0,
#  'JYcCYNWs8Ul6ewG5kCYW4Q': 1,
#  'woFthCAsX2JYi8i2qEBb1w': 2,
# }
# `
# 
# ### bid2idx_dict:
# keeps the mapping relation between unique businessid to an business index, which is in this format:\
# `
# {
#  '3xykzfVY2PbdjKCRDLdzTQ': 0,
#  'R7-Art-yi73tWaRTuXXH7w': 1,
#  'DYuOxkW4DtlJsTHdxdXSlg': 2,
# }
# `

jsonLine_rdd = sc.textFile(train_file_path).map(lambda line:json.loads(line)).persist()

# read in the user/bus mapping dict
use_test_gt = True
if use_test_gt:
    uid_index_train_rdd = jsonLine_rdd.map(lambda jLine:jLine['user_id']).distinct()
    uid_index_gt_rdd = sc.textFile(test_groundtruth_path)                        .map(lambda line:json.loads(line))                        .map(lambda jLine:jLine['user_id']).distinct()
    uid_index_rdd = uid_index_train_rdd.union(uid_index_gt_rdd).distinct().zipWithIndex()
    
    bid_index_train_rdd = jsonLine_rdd.map(lambda jLine:jLine['business_id']).distinct()
    bid_index_gt_rdd = sc.textFile(test_groundtruth_path)                        .map(lambda line:json.loads(line))                        .map(lambda jLine:jLine['business_id']).distinct()
    bid_index_rdd = bid_index_train_rdd.union(bid_index_gt_rdd).distinct().zipWithIndex()
else:
    uid_index_rdd = jsonLine_rdd.map(lambda jLine:jLine['user_id']).distinct().zipWithIndex()
    bid_index_rdd = jsonLine_rdd.map(lambda jLine:jLine['business_id']).distinct().zipWithIndex()
    
uid2idx_dict = dict(uid_index_rdd.collect())
bid2idx_dict = dict(bid_index_rdd.collect())

def outputDictModel(id2idx_dict, path):
    model_writer = open(path, 'w')
    for ub_id in id2idx_dict:
        line = str(ub_id) + ', ' + str(id2idx_dict[ub_id]) + '\n'
        model_writer.write(line)
    model_writer.close()
    
outputDictModel(uid2idx_dict, uid2idx_output_path)
outputDictModel(bid2idx_dict, bid2idx_output_path)


def getKnownFriendsList(friends_list):
    known_friends_idx = list()
    for friend in friends_list:
        fidx = str(uid2idx_dict.get(friend, 'UNK'))
        if fidx != 'UNK':
            known_friends_idx.append(fidx)
    return known_friends_idx
    

# friends list
friends = (sc.textFile(friends_path)
           .map(lambda line:json.loads(line))
           .map(lambda jLine:(jLine['user_id'], jLine['friends']))
           .map(lambda pair: (uid2idx_dict.get(pair[0], pair[0]), pair[1].split(', ')))
           .map(lambda pair:(str(pair[0]), getKnownFriendsList(pair[1])))
           .collectAsMap()
          )

model_writer = open(friends_output_path, 'w')
for user in friends:
    jLine = json.dumps({'u':user, 'friends':friends[user]})
    model_writer.write(jLine + '\n')
model_writer.close()


user_avg_dict = (sc.textFile(user_avg_file)
                 .map(lambda line: json.loads(line))
                 .flatMap(lambda kv_items: kv_items.items()) 
                 .map(lambda pair: (uid2idx_dict.get(pair[0], pair[0]), pair[1]))
                 .collectAsMap()
                )

bus_avg_dict = (sc.textFile(bus_avg_file)
                 .map(lambda line: json.loads(line))
                 .flatMap(lambda kv_items: kv_items.items()) 
                 .map(lambda pair: (bid2idx_dict.get(pair[0], pair[0]), pair[1]))
                 .collectAsMap()
                )

ALL_USER_AVG_RATING = sum(user_avg_dict.values())/len(user_avg_dict)
ALL_BUS_AVG_RATING = sum(bus_avg_dict.values())/len(bus_avg_dict)


# build the SVD algorithm using sklearn surprise
uidx_bidx_star = (jsonLine_rdd.map(lambda jLine: (uid2idx_dict[jLine["user_id"]],
                                                  bid2idx_dict[jLine["business_id"]], 
                                                  jLine["stars"]
                                                 )
                                  )
                  .collect()
                 )

random.seed(511)
for _ in range(300000):
    uidx = random.sample(uidx_bidx_star, 1)[0][0]
    bidx = random.sample(uidx_bidx_star, 1)[0][1]
    rating = user_avg_dict.get(uidx, bus_avg_dict.get(bidx, ALL_BUS_AVG_RATING))
    up_sampling_pair = (uidx, bidx, rating)
    uidx_bidx_star.append(up_sampling_pair)


train_data_df = pd.DataFrame(uidx_bidx_star, columns=['uidx', 'bidx', 'rating'])

reader = Reader()
surprise_train_dataset = Dataset.load_from_df(train_data_df[['uidx', 'bidx', 'rating']], reader)
trainset = surprise_train_dataset.build_full_trainset()

svd_model = SVD(n_factors=10, biased=True, init_std_dev=0.05)
svd_model.fit(trainset)

dump.dump(SVD_model_output_path, predictions=None, algo=svd_model)


feature_names = ['uavg', 'bavg', 'useful', 'funny', 'cool', 'fans', 'elite', 'user_num', 'bus_num', 'rating']
default_tuple = (0,0,0,0,0)

user_info_dict = (sc.textFile(user_file_path)
                 .map(lambda line:json.loads(line))
                 .map(lambda jLine:(uid2idx_dict.get(jLine['user_id'], jLine['user_id']), 
                                     (jLine['useful'], jLine['funny'], jLine['cool'], jLine['fans'],
                                      len(jLine['elite'].split(','))
                                     )))
                 .collectAsMap()
                )


uidx_star_dict = (sc.textFile(train_file_path).map(lambda line:json.loads(line))
                       .map(lambda jLine:(jLine['user_id'], jLine['stars']))
                       .map(lambda pair:(str(uid2idx_dict[pair[0]]), pair[1]))
                       .groupByKey()
                       .mapValues(lambda listOfTuple:list(listOfTuple))
                       .collectAsMap()
                      )

bidx_star_dict = (sc.textFile(train_file_path).map(lambda line:json.loads(line))
                       .map(lambda jLine:(jLine['business_id'], jLine['stars']))
                       .map(lambda pair:(str(bid2idx_dict[pair[0]]), pair[1]))
                       .groupByKey()
                       .mapValues(lambda listOfTuple:list(listOfTuple))
                       .collectAsMap()
                      )


def getUserInfo(pair):
    user = pair[0]
    bus = pair[1]
    user_avg = user_avg_dict.get(user, ALL_USER_AVG_RATING)
    bus_avg = bus_avg_dict.get(bus, ALL_BUS_AVG_RATING)
    feature_list = [user_avg, bus_avg]
    
    info_tuple = user_info_dict.get(user, default_tuple)
    for feature in info_tuple:
        feature_list.append(feature) 

    user_num_rating = len(uidx_star_dict.get(user, []))
    bus_num_rating = len(bidx_star_dict.get(bus, []))
    feature_list.append(user_num_rating)
    feature_list.append(bus_num_rating)
    feature_list.append(pair[2]) # the rating
    
    return tuple(feature_list)

uidx_bidx_star_xgb = (jsonLine_rdd.map(lambda jLine: (uid2idx_dict[jLine["user_id"]],
                                                      bid2idx_dict[jLine["business_id"]], 
                                                      jLine["stars"]
                                                     )
                                      )
                      .collect()
                     )

if use_test_gt:
    gt_uidx_bidx_star_xgb = (sc.textFile(test_groundtruth_path).map(lambda line:json.loads(line))
                            .map(lambda jLine: (uid2idx_dict.get(jLine["user_id"], jLine["user_id"]),
                                                bid2idx_dict.get(jLine["business_id"], jLine["business_id"]), 
                                                jLine["stars"]
                                                )
                                          )
                          .collect()
                         )
    uidx_bidx_star_xgb.extend(gt_uidx_bidx_star_xgb)


# convert uidx and bidx to avg rating and keep the original rating
xgboost_training_data = list()
for pair in uidx_bidx_star_xgb:
    train_tuple = getUserInfo(pair)
    xgboost_training_data.append(train_tuple)

train_data_df = pd.DataFrame(xgboost_training_data, columns=feature_names)


model = xgboost.XGBRegressor(booster='gbtree', max_depth=8, eta=0.47)
model.fit(X=train_data_df.iloc[:,:-1], y=train_data_df.iloc[:,-1])


pickle.dump(model, open(xgb_model_output_path, 'wb'))


print('Duration:', str(time.time()-startTime))




