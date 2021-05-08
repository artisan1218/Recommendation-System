#!/usr/bin/env python
# coding: utf-8
from pyspark import SparkContext, SparkConf
import sys
import json
import os
import pandas as pd
import numpy as np
import time
from surprise import SVD, Reader, Dataset, dump
import xgboost
import pickle

startTime = time.time()

os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'

conf = SparkConf().setMaster("local[*]").set("spark.executor.memory", "4g").set("spark.driver.memory", "4g")
sc = SparkContext(conf=conf).getOrCreate()


train_file_path = "../resource/asnlib/publicdata/train_review.json"
test_file_path = sys.argv[1] #r'C:\Users\11921\Downloads\competition\test_review.json'
output_file_path = sys.argv[2] #r'hybrid_rec_sys_pred.txt'

test_groundtruth_path = "./test_review_ratings.json"
SVD_model_path = "./svd_model.model"
xgb_model_path = "./xgb_model.json"
user_avg_file = "../resource/asnlib/publicdata/user_avg.json"
bus_avg_file = "../resource/asnlib/publicdata/business_avg.json"
user_file_path = "../resource/asnlib/publicdata/user.json"
friends_file = "./friends_model.json"

uid2idx_path = "./uid2idx_dict_model.txt"
bid2idx_path = "./bid2idx_dict_model.txt"


# read in the user/bus mapping dict
uid2idx_dict = sc.textFile(uid2idx_path).map(lambda line: line.split(', ')).collectAsMap()
bid2idx_dict = sc.textFile(bid2idx_path).map(lambda line: line.split(', ')).collectAsMap()
uidx2id_dict = {v: k for k, v in uid2idx_dict.items()}
bidx2id_dict = {v: k for k, v in bid2idx_dict.items()}

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

friends_dict = (sc.textFile(friends_file)
                .map(lambda line:json.loads(line))
                .map(lambda jLine:(jLine['u'], jLine['friends']))
                .collectAsMap()
               )



feature_names = ['uavg', 'bavg', 'useful', 'funny', 'cool', 'fans', 'elite', 'user_num', 'bus_num']
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
                       .map(lambda pair:(uid2idx_dict[pair[0]], pair[1]))
                       .groupByKey()
                       .mapValues(lambda listOfTuple:list(listOfTuple))
                       .collectAsMap()
                      )

bidx_star_dict = (sc.textFile(train_file_path).map(lambda line:json.loads(line))
                       .map(lambda jLine:(jLine['business_id'], jLine['stars']))
                       .map(lambda pair:(bid2idx_dict[pair[0]], pair[1]))
                       .groupByKey()
                       .mapValues(lambda listOfTuple:list(listOfTuple))
                       .collectAsMap()
                      )

uidx_star_dict_gt = (sc.textFile(test_groundtruth_path).map(lambda line:json.loads(line))
                       .map(lambda jLine:(jLine['user_id'], jLine['stars']))
                       .map(lambda pair:(uid2idx_dict[pair[0]], pair[1]))
                       .groupByKey()
                       .mapValues(lambda listOfTuple:list(listOfTuple))
                       .collectAsMap()
                      )

bidx_star_dict_gt = (sc.textFile(test_groundtruth_path).map(lambda line:json.loads(line))
                       .map(lambda jLine:(jLine['business_id'], jLine['stars']))
                       .map(lambda pair:(bid2idx_dict[pair[0]], pair[1]))
                       .groupByKey()
                       .mapValues(lambda listOfTuple:list(listOfTuple))
                       .collectAsMap()
                      )

svd_model = dump.load(SVD_model_path)[1]

xgb_model = pickle.load(open(xgb_model_path, 'rb'))

ALL_USER_AVG_RATING = sum(user_avg_dict.values())/len(user_avg_dict)
ALL_BUS_AVG_RATING = sum(bus_avg_dict.values())/len(bus_avg_dict)


def secondaryPred(uidx, bidx, friends_rating_method):
    friends = friends_dict.get(uidx, [])
    if len(friends)!=0:
        predicted_rating_list = list()
        for friend in friends:
            f_rating_dict = dict(uidx_bidx_star_dict.get(friend, {})) # get user's friends's all ratings
            if friends_rating_method == 'avg':
                if len(f_rating_dict)!=0:
                    f_avg_rating = sum(f_rating_dict.values()) / len(f_rating_dict)
                    predicted_rating_list.append(f_avg_rating)
            elif friends_rating_method == 'bidx':
                f_rating = f_rating_dict.get(bidx, -1)
                # his friend has rated this bidx before
                if f_rating != -1: 
                    predicted_rating_list.append(f_rating)

        # this user has no friends or less than at least 5 of his friends rated this bus before
        if len(predicted_rating_list)==0:
            return (uidx, bidx, bus_avg_dict.get(bidx, ALL_BUS_AVG_RATING))
        else:
            all_f_avg_rating = sum(predicted_rating_list)/len(predicted_rating_list)
            #bidx_avg = bus_avg_dict.get(bidx, BUS_AVG_RATING)
            #return (uidx, bidx, 0.2*all_f_avg_rating + 0.8*bidx_avg)
            return (uidx, bidx, all_f_avg_rating)
    else:
        # this user has not rated anything and he has no friendS
        return (uidx, bidx, bus_avg_dict.get(bidx, ALL_BUS_AVG_RATING))

        
def svd_predict(ub_pair, includeSecPred=True):
    uidx = ub_pair[0]
    bidx = ub_pair[1]
    
    if includeSecPred == True:
        # both uidx and bidx are seen,use SVD first
        if isinstance(uidx, int) and isinstance(bidx, int):
            # prediction result by svd model
            prediction = svd_model.predict(uid=uidx, iid=bidx)[3] # the third field is predicted score
            return (uidx, bidx, prediction)
        else: 
            '''either uidx is unseen, bidx is unseen or both are unseen'''
            if isinstance(uidx, int) and isinstance(bidx, str):
                # uidx is seen, but bidx is unseen, bidx is not in the train_review.json
                '''make prediction through the known user friends?'''
                # convert int to str so that they can be used by other models
                uidx = str(uidx)
                prediction_method = 'friends' # avg
                if prediction_method == 'avg':
                    # use the avg rating of this uidx
                    prediction = user_avg_dict.get(uidx, ALL_USER_AVG_RATING)
                    return (uidx, bidx, prediction)
                elif prediction_method=='friends':
                    return secondaryPred(uidx, bidx, friends_rating_method='avg') 
            elif isinstance(bidx, int) and isinstance(uidx, str):
                # bidx is seen, but uidx is unseen, uidx is not in the train_review.json
                # but his friends might be in the train_review.json
                bidx = str(bidx)
                return secondaryPred(uidx, bidx, friends_rating_method = 'avg')
            else:
                # both are unseen
                # convert int to str so that they can be used by other models
                uidx = str(uidx)
                bidx = str(bidx)
                return (uidx, bidx, 0.5*ALL_USER_AVG_RATING + 0.5*ALL_BUS_AVG_RATING)
    else:
        # use only svd model to predict
        uidx = str(uidx)
        bidx = str(bidx)
        prediction = svd_model.predict(uid=uidx, iid=bidx)[3] # the third field is predicted score
        return (uidx, bidx, prediction)
    
def getUserInfo(pair):
    user = pair[0]
    bus = pair[1]
    user_avg = user_avg_dict.get(user, ALL_USER_AVG_RATING)
    bus_avg = bus_avg_dict.get(bus, ALL_BUS_AVG_RATING)
    feature_list = [user_avg, bus_avg]
    
    info_tuple = user_info_dict.get(user, default_tuple)
    for feature in info_tuple:
        feature_list.append(feature) 

    #user_num_rating = len(uidx_star_dict.get(pair[0], uidx_star_dict_gt.get(pair[0], [])))
    #bus_num_rating = len(bidx_star_dict.get(pair[1], bidx_star_dict_gt.get(pair[1], [])))
    
    user_num_rating = len(uidx_star_dict.get(pair[0], []))
    bus_num_rating = len(bidx_star_dict.get(pair[1], []))
    
    feature_list.append(user_num_rating)
    feature_list.append(bus_num_rating)
    
    return tuple(feature_list)
        
        
def xgb_predict(model, data_input, includeSecPred=True):
    if includeSecPred == True:
        xgb_predict_input = list()
        xgb_predict_ub_idx = list()
        known_uidx_unknown_bidx_list = list()
        unknown_uidx_known_bidx_list = list()
        unknown_uidx_bidx_list = list()
        for pair in data_input:
            uidx = pair[0]
            bidx = pair[1]
            if isinstance(uidx, int) and isinstance(bidx, int):
                uidx = str(uidx)
                bidx = str(bidx)
                predict_pair = getUserInfo((uidx, bidx))
                xgb_predict_input.append(predict_pair)
                xgb_predict_ub_idx.append((uidx, bidx))
            elif isinstance(uidx, int) and isinstance(bidx, str):
                # seen uidx, unseen bidx
                known_uidx_unknown_bidx_list.append((str(uidx), bidx))
            elif isinstance(bidx, int) and isinstance(uidx, str):
                # seen bidx, unseen uidx
                unknown_uidx_known_bidx_list.append((uidx, str(bidx)))
            elif isinstance(uidx, str) and isinstance(bidx, str):
                # both are unseen
                unknown_uidx_bidx_list.append((uidx, bidx))
        
        # predict the rating for all known pairs
        predict_data_df = pd.DataFrame(xgb_predict_input, columns=feature_names)
        xgb_prediction = model.predict(predict_data_df)
        
        # put the predicted rating back to prediction list to output
        prediction_list = list()
        for pair in zip(xgb_predict_ub_idx, xgb_prediction):
            # pair[0][0] is uidx, pair[0][1] is bidx, pair[1] is the predicted rating
            prediction_list.append((pair[0][0], pair[0][1], pair[1]))
        
        # now overwrite the prediction result of the unknwon pair, this way we can perserve the order of prediction
        for pair in known_uidx_unknown_bidx_list:
            uidx = pair[0]
            bidx = pair[1]
            prediction_method = 'friends' # avg
            if prediction_method == 'avg':
                # use the avg rating of this uidx
                prediction = user_avg_dict.get(uidx, ALL_USER_AVG_RATING)
            elif prediction_method=='friends':
                prediction = secondaryPred(uidx, bidx, friends_rating_method='avg')[2]
            prediction_list.append((uidx, bidx, prediction))
        
        for pair in unknown_uidx_known_bidx_list:
            uidx = pair[0]
            bidx = pair[1]
            prediction = secondaryPred(uidx, bidx, friends_rating_method='avg')[2]
            prediction_list.append((uidx, bidx, prediction))
            
        for pair in unknown_uidx_bidx_list:
            prediction = 0.5*ALL_USER_AVG_RATING + 0.5*ALL_BUS_AVG_RATING
            prediction_list.append((pair[0], pair[1], prediction))
            
        return prediction_list
    else:
        xgb_predict_input = list()
        # convert the pair into avg user and bus rating to use xgboost model
        for pair in data_input:
            uidx = str(pair[0])
            bidx = str(pair[1])
            predict_pair = getUserInfo((uidx, bidx))
            xgb_predict_input.append(predict_pair)
            
        # predict the rating
        predict_data_df = pd.DataFrame(xgb_predict_input, columns=feature_names)
        xgb_prediction = model.predict(predict_data_df)

        # put the predicted rating back to prediction list to output
        prediction = list()
        for pair in zip(data_input, xgb_prediction):
            # pair[0][0] is uidx, pair[0][1] is bidx, pair[1] is the predicted rating
            prediction.append((pair[0][0], pair[0][1], pair[1]))
        return prediction


# ### predict_uidx_bidx
# is in this format: (uidx, bidx)\
# `
# [
#  ('14173', '2090'),
#  ('14264', '4573'),
#  ('14270', '9538')
#  ...
# ]
# `\
# These are the list of uidx, bidx pairs that we're going to predict


# turn idx into int instead of str, so that we can easily distingush known and unknown id
uid2idx_dict = dict([(pair[0], int(pair[1])) for pair in uid2idx_dict.items()]) 
bid2idx_dict = dict([(pair[0], int(pair[1])) for pair in bid2idx_dict.items()]) 


predict_uidx_bidx_rdd = (sc.textFile(test_file_path)
                         .map(lambda x:json.loads(x))
                         .map(lambda jLine: (jLine['user_id'], jLine['business_id']))
                         .map(lambda pair:(uid2idx_dict.get(pair[0], pair[0]), bid2idx_dict.get(pair[1], pair[1])))
                         .persist()
                        )

# uidx_bidx_star_dict is used to see if a user has rated certain bus before
uidx_bidx_star_dict = (sc.textFile(train_file_path).map(lambda line:json.loads(line))
                       .map(lambda jLine:(jLine['user_id'], jLine['business_id'], jLine['stars']))
                       .map(lambda pair:(str(uid2idx_dict[pair[0]]), (str(bid2idx_dict[pair[1]]), pair[2])))
                       .groupByKey()
                       .mapValues(lambda listOfTuple:list(listOfTuple))
                       .collectAsMap()
                      )


model = 'mixed' # svd, xgb, mixed

if model == 'svd': 
    prediction = predict_uidx_bidx_rdd.map(lambda ub_pair: svd_predict(ub_pair, includeSecPred=True)).collect()
elif model == 'xgb':
    # get the predicting pair
    predict_ub_idx = predict_uidx_bidx_rdd.collect()
    
    xgboost_test_data = list()
    xgboost_test_y = list()
    for pair in predict_ub_idx:
        test_tuple = getUserInfo((str(pair[0]), str(pair[1])))
        xgboost_test_data.append(test_tuple)
    test_X_df = pd.DataFrame(xgboost_test_data, columns=feature_names)
    xgboost_prediction = xgb_model.predict(test_X_df)
    
    prediction = list()
    for pair in zip(predict_ub_idx, xgboost_prediction):
        # pair[0][0] is uidx, pair[0][1] is bidx, pair[1] is the predicted rating
        prediction.append((pair[0][0], pair[0][1], pair[1]))
    
    #prediction = xgb_predict(xgb_model, predict_ub_idx, includeSecPred=False)
elif model == 'mixed':
    # mixed model will take the weighted average of both svd and xgb model
    svd_prediction = predict_uidx_bidx_rdd.map(lambda ub_pair: svd_predict(ub_pair, includeSecPred=True)).collect()
    
    predict_ub_idx = predict_uidx_bidx_rdd.collect()
    xgb_prediction = xgb_predict(xgb_model, predict_ub_idx, includeSecPred=False)
      
    prediction = list()
    for pair in zip(svd_prediction, xgb_prediction):
        svd = pair[0]
        xgb = pair[1]
        mixed_rating = min(0.15*float(svd[2]) + 0.85*float(xgb[2]), 5.0)
        prediction.append((svd[0], svd[1], mixed_rating))
    

resultWriter = open(output_file_path, 'w')
for pair in prediction:
    uid = uidx2id_dict.get(str(pair[0]), str(pair[0]))
    bid = bidx2id_dict.get(str(pair[1]), str(pair[1]))
    jsonPair = {'user_id':uid, 'business_id':bid, 'stars':float(pair[2])}
    resultWriter.write(json.dumps(jsonPair) + '\n')
resultWriter.close()


print('Duration:', str(time.time()-startTime))







