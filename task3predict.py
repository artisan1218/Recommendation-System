from pyspark import SparkContext, SparkConf
import json
import math
import time
import sys

startTime = time.time()

train_file = sys.argv[1] #r'C:\Users\11921\Downloads\data\train_review.json'
test_file = sys.argv[2] #r'C:\Users\11921\Downloads\data\test_review.json'
model_file = sys.argv[3]
output_file = sys.argv[4]
cf_type = sys.argv[5]

conf = SparkConf().setMaster("local[*]")         .set("spark.executor.memory", "4g")         .set("spark.driver.memory", "4g")
sc = SparkContext(conf=conf).getOrCreate()



jsonLine_rdd = sc.textFile(train_file).map(lambda line:json.loads(line))
uid_index_rdd = jsonLine_rdd.map(lambda jsonLine:jsonLine['user_id']).collect()
bid_index_rdd = jsonLine_rdd.map(lambda jsonLine:jsonLine['business_id']).collect()
uid2idx_dict = dict()
bid2idx_dict = dict()
index = 0
for uid in uid_index_rdd:
    if uid not in uid2idx_dict.keys():
        uid2idx_dict[uid] = index
    index+=1
    
index = 0
for bid in bid_index_rdd:
    if bid not in bid2idx_dict.keys():
        bid2idx_dict[bid] = index
    index+=1


def processSameUserSameBusMultiStars(pair):
    bidx_star_dict = dict()
    uidx = pair[0]
    bidx_star_pairList = pair[1]
    bus_avgStar_list = list()
    for bidx_star in bidx_star_pairList:
        if bidx_star[0] not in bidx_star_dict.keys():
            bidx_star_dict[bidx_star[0]] = [bidx_star[1]]
        else:
            bidx_star_dict[bidx_star[0]].append(bidx_star[1])
    for bidx, stars in bidx_star_dict.items():
        avgStar = sum(stars)/len(stars)
        bus_avgStar_list.append((bidx, avgStar))
    return (uidx, bus_avgStar_list)

def userBasedPred(pair):
    uidx = pair[0]
    bidx = pair[1]
    
    if uidx in uidx_bidx_star_dict.keys():
        uidx_all_rates = dict(uidx_bidx_star_dict[uidx]).values()
        r = sum(uidx_all_rates) / len(uidx_all_rates)
        numerator = 0
        denumerator = 0
        # Use top N pearson cor that had rated bidx to copmute prediction
        neighbors_list = user_model_dict.get(uidx, [])
        if len(neighbors_list)==0:
            #cold start, the user has no corresponding pearson correlation calculated in model
            #use avg stars of the bus as the prediction
            prediction = sum(bidx_stars_dict[bidx]) / len(bidx_stars_dict[bidx])
            return (pair[0], pair[1], prediction)
        else:
            # choose top N neighbor
            top_N = 5
            neighbors_counter = 0
            for neighbor in neighbors_list:
                uidx2 = neighbor[0]
                pearson = neighbor[1]
                # check if uidx2 had rated bidx, otherwise skip
                bidx_star_dict = dict(uidx_bidx_star_dict[uidx2])
                if bidx in bidx_star_dict.keys():
                    bidx_rate_for_uidx2 = bidx_star_dict[bidx]
                    avg_rate_uidx2 = (sum(bidx_star_dict.values()) - bidx_rate_for_uidx2) / (len(bidx_star_dict) - 1)
                    numerator += (bidx_rate_for_uidx2 - avg_rate_uidx2) * pearson
                    denumerator += pearson
                    neighbors_counter += 1
                    if neighbors_counter >= top_N:
                        break
            if denumerator == 0:
                uidx_all_rates = dict(uidx_bidx_star_dict[uidx]).values()
                prediction = sum(uidx_all_rates) / len(uidx_all_rates)
            else:
                prediction = r + numerator/denumerator
            return (pair[0], pair[1], prediction)
    else:
        prediction = sum(bidx_stars_dict[bidx]) / len(bidx_stars_dict[bidx])
        return (uidx, bidx, prediction)
        
def itemBasedPred(pair):
    uidx = pair[0]
    bidx = pair[1]
    
    # Use top N pearson cor that had rated bidx to copmute prediction
    neighbors_list = item_model_dict.get(bidx, [])
    if len(neighbors_list) == 0:
        #cold start
        bidx_star_dict = dict(uidx_bidx_star_dict[uidx])
        prediction = sum(bidx_star_dict.values()) / len(bidx_star_dict)
        return (uidx, bidx, prediction)
    else:
        top_N = 5
        neighbors_counter = 0
        numerator = 0
        denumerator = 0
        for neighbor in neighbors_list:
            bidx2 = neighbor[0]
            pearson = neighbor[1]
            bidx_star_dict = dict(uidx_bidx_star_dict[uidx])
            # check if this bus had been rated by this user
            if bidx2 in bidx_star_dict.keys():
                numerator += pearson * bidx_star_dict[bidx2]
                denumerator += pearson
                neighbors_counter += 1
                if neighbors_counter >= top_N:
                    break
        if denumerator == 0:
            bidx_star_dict = dict(uidx_bidx_star_dict[uidx])
            prediction = sum(bidx_star_dict.values()) / len(bidx_star_dict)
            return (uidx, bidx, prediction)
        else:
            prediction = numerator/denumerator
        return (uidx, bidx, prediction)


if cf_type == 'user_based':
    predict_uidx_bidx_pairs = sc.textFile(test_file, 12).map(lambda x:json.loads(x))                    .map(lambda jsonLine:(jsonLine['user_id'], jsonLine['business_id']))                    .map(lambda pair:(uid2idx_dict.get(pair[0], -1), bid2idx_dict.get(pair[1], -1)))                    .filter(lambda pair: pair[0]!=-1 and pair[1]!=-1)
                    
    user_model1 = sc.textFile(model_file).map(lambda x:json.loads(x))                                   .map(lambda jsonLine:(jsonLine['u1'], jsonLine['u2'], jsonLine['sim']))                                   .map(lambda pair:(uid2idx_dict[pair[0]], (uid2idx_dict[pair[1]], pair[2])))
    user_model_dict = sc.textFile(model_file).map(lambda x:json.loads(x))                                   .map(lambda jsonLine:(jsonLine['u2'], jsonLine['u1'], jsonLine['sim']))                                   .map(lambda pair:(uid2idx_dict[pair[0]], (uid2idx_dict[pair[1]], pair[2])))                                   .union(user_model1).groupByKey().mapValues(lambda x:list(x))                                   .mapValues(lambda l:sorted(l, key=lambda x:-x[1])).collectAsMap()
                                    
    uidx_bidx_star_dict = jsonLine_rdd.map(lambda jLine:(jLine['user_id'], jLine['business_id'], jLine['stars']))                                 .map(lambda pair:(uid2idx_dict[pair[0]], (bid2idx_dict[pair[1]], pair[2])))                                 .groupByKey().mapValues(lambda listOfTuple:list(listOfTuple))                                 .map(lambda pair:processSameUserSameBusMultiStars(pair))                                 .filter(lambda pair:len(pair[1])>=3).collectAsMap()
                                
    bidx_stars_dict = jsonLine_rdd.map(lambda jLine:(bid2idx_dict[jLine['business_id']], jLine['stars']))                                  .groupByKey().mapValues(lambda starsList:list(starsList)).collectAsMap()
    star_prediction = predict_uidx_bidx_pairs.map(lambda pair: userBasedPred(pair)).collect()
    
    uidx2uid_dict = {v: k for k, v in uid2idx_dict.items()}
    bidx2bid_dict = {v: k for k, v in bid2idx_dict.items()}
    
    resultWriter = open(output_file, 'w')
    for pair in star_prediction:
        uid = uidx2uid_dict[pair[0]]
        bid = bidx2bid_dict[pair[1]]
        jsonPair = {'user_id':uid, 'business_id':bid, 'stars':pair[2]}
        resultWriter.write(json.dumps(jsonPair) + '\n')
    resultWriter.close()
    
else:
    predict_uidx_bidx_pairs = sc.textFile(test_file, 100).map(lambda x:json.loads(x))                    .map(lambda jsonLine:(jsonLine['user_id'], jsonLine['business_id']))                    .map(lambda pair:(uid2idx_dict.get(pair[0], -1), bid2idx_dict.get(pair[1], -1)))                    .filter(lambda pair: pair[0]!=-1 and pair[1]!=-1)
    
    item_model1 = sc.textFile(model_file).map(lambda x:json.loads(x))                                   .map(lambda jsonLine:(jsonLine['b1'], jsonLine['b2'], jsonLine['sim']))                                   .map(lambda pair:(bid2idx_dict[pair[0]], (bid2idx_dict[pair[1]], pair[2]))).cache()
    item_model_dict = dict(item_model1.map(lambda pair: (pair[1][0], (pair[0], pair[1][1])))                                   .union(item_model1).groupByKey().mapValues(lambda x:list(x))                                   .mapValues(lambda l:sorted(l, key=lambda x:-x[1])).collect())
    
    uidx_bidx_star_dict = dict(jsonLine_rdd.map(lambda jLine:(jLine['user_id'], jLine['business_id'], jLine['stars']))                                 .map(lambda pair:(uid2idx_dict[pair[0]], (bid2idx_dict[pair[1]], pair[2])))                                 .groupByKey().mapValues(lambda listOfTuple:list(listOfTuple)).collect())
                                 
 
    star_prediction = predict_uidx_bidx_pairs.map(lambda pair: itemBasedPred(pair)).collect()

    uidx2uid_dict = {v: k for k, v in uid2idx_dict.items()}
    bidx2bid_dict = {v: k for k, v in bid2idx_dict.items()}

    resultWriter = open(output_file, 'w')
    for pair in star_prediction:
        uid = uidx2uid_dict[pair[0]]
        bid = bidx2bid_dict[pair[1]]
        jsonPair = {'user_id':uid, 'business_id':bid, 'stars':pair[2]}
        resultWriter.write(json.dumps(jsonPair) + '\n')
    resultWriter.close()

print('Duration:', str(time.time()-startTime))





