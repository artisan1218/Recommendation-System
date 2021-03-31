

from pyspark import SparkContext, SparkConf
import itertools
from collections import Counter
import json
import math
import random
import time
import sys
import string



startTime = time.time()
input_file_path = sys.argv[1]#r'C:\Users\11921\Downloads\data\train_review.json'
model_file_path = sys.argv[2]
cf_type = sys.argv[3]
# In[3]:


conf = SparkConf().setMaster("local[*]")         .set("spark.executor.memory", "4g")         .set("spark.driver.memory", "4g")
sc = SparkContext(conf=conf).getOrCreate()


# In[4]:


jsonLine_rdd = sc.textFile(input_file_path).map(lambda line:json.loads(line))


# In[5]:


# Create id to index mapping in form of dict for user and businesss
# Convert id to index so that simliarity is easy to be computed
uid_index_rdd = jsonLine_rdd.map(lambda jsonLine:jsonLine['user_id']).distinct().zipWithIndex()
bid_index_rdd = jsonLine_rdd.map(lambda jsonLine:jsonLine['business_id']).distinct().zipWithIndex()
uid_index_dict = uid_index_rdd.collectAsMap()
bid_index_dict = bid_index_rdd.collectAsMap()


# In[6]:


def processSameBusSameUserMultiStars(busPair):
    user_star_dict = dict()
    bidx = busPair[0]
    user_star_list = busPair[1]
    user_avgStar_list = list()
    for user_star in user_star_list:
        uidx = user_star[0]
        star = user_star[1]
        if uidx not in user_star_dict.keys():
            user_star_dict[uidx] = [star]
        else:
            user_star_dict[uidx].append(star)
    for user, stars in user_star_dict.items():
        avgStar = sum(stars)/len(stars)
        user_avgStar_list.append((user, avgStar))
    return (bidx, user_avgStar_list)

def findCandidatesPair(pair):
    bidx1 = pair[0]
    bidx2 = pair[1]
    uidx1_star_list = bidx_uidx_star_dict[bidx1]
    uidx2_star_list = bidx_uidx_star_dict[bidx2]
    uidx1_list = [user_star[0] for user_star in uidx1_star_list]
    uidx2_list = [user_star[0] for user_star in uidx2_star_list]
    corated_num = len(set(uidx1_list).intersection(set(uidx2_list)))
    return (pair, corated_num)

def pearsonCorrelation(pair):
    bidx1 = pair[0]
    bidx2 = pair[1]
    user_rates_dict1 = dict(bidx_uidx_star_dict[bidx1])
    user_rates_dict2 = dict(bidx_uidx_star_dict[bidx2])
    
    '''
    # Average rating using all ratings
    bidx1_rates_list = user_rates_dict1.values()
    bidx2_rates_list = user_rates_dict2.values()
    bidx1_avgRate = sum(bidx1_rates_list) / len(bidx1_rates_list)
    bidx2_avgRate = sum(bidx2_rates_list) / len(bidx2_rates_list)
    '''
    # Average rating using only corated ratings
    corated_users_list = list(set(user_rates_dict1.keys()).intersection(set(user_rates_dict2.keys())))
    bidx1_corated_rating = [user_rates_dict1[corated_user] for corated_user in corated_users_list]
    bidx2_corated_rating = [user_rates_dict2[corated_user] for corated_user in corated_users_list]
    bidx1_avgRate = sum(bidx1_corated_rating) / len(bidx1_corated_rating)
    bidx2_avgRate = sum(bidx2_corated_rating) / len(bidx2_corated_rating)
    
    numerator = 0
    denumerator1 = 0
    denumerator2 = 0
    for corated_user in corated_users_list:
        numerator += (user_rates_dict1[corated_user] - bidx1_avgRate) * (user_rates_dict2[corated_user] - bidx2_avgRate)
        denumerator1 += (user_rates_dict1[corated_user] - bidx1_avgRate)**2
        denumerator2 += (user_rates_dict2[corated_user] - bidx2_avgRate)**2
    if denumerator1 == 0 or denumerator2 == 0:
        pearson = 0
    else:
        pearson = numerator / (math.sqrt(denumerator1) * math.sqrt(denumerator2))
    return (bidx1, bidx2, pearson)

#below is to compute user based

def minhash(row, num_bins):
    uidx = row[0]
    bidx_list = row[1]
    minhash_signature_list = list()
    for params in hash_func_list: 
        hashed_bidx_list = list() 
        param_a = params[0]
        param_b = params[1]
        param_p = params[2]
        param_m = num_bins
        for bidx in bidx_list: # hash all the bidx using each hash function and find minhash signature for that hash func
            hashed_bidx = ((param_a*int(bidx)+param_b)%param_p)%param_m
            hashed_bidx_list.append(hashed_bidx)
        minhash_signature = min(hashed_bidx_list) # 1 of the 50 minhash signature for this user
        minhash_signature_list.append(minhash_signature)
    return (uidx, minhash_signature_list)

def createBands(sig, band_num):
    uidx = sig[0]
    sig_list = sig[1]
    band_list = list()
    row_num = int(num_hash_func/band_num) # in this case: 50/50 = 1
    for i in range(band_num):
        band_list.append(((i, hash(tuple(sig_list[i * row_num: i * row_num + row_num]))), uidx))
    return band_list

def computeJaccard(can_tuple):
    can1 = can_tuple[0]
    can2 = can_tuple[1]
    busList1 = set(char_matrix[can1])
    busList2 = set(char_matrix[can2])
    jaccard = len(busList1.intersection(busList2))/len(busList1.union(busList2))
    return (can1, can2, jaccard)

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

def computePearson(pair):
    uidx1 = pair[0]
    uidx2 = pair[1]
    bidx_dict1 = dict(uidx_bidx_star_dict.get(uidx1, []))
    bidx_dict2 = dict(uidx_bidx_star_dict.get(uidx2, []))
    corated_list = list(set(bidx_dict1.keys()).intersection(set(bidx_dict2.keys())))
    
    if len(corated_list)>=3:
        # compute pearson
        corated_star1 = [bidx_dict1[corated] for corated in corated_list]
        corated_star2 = [bidx_dict2[corated] for corated in corated_list]
        avg1 = sum(corated_star1)/len(corated_star1)
        avg2 = sum(corated_star2)/len(corated_star2)
        numerator = 0
        denumerator1 = 0
        denumerator2 = 0
        for corated in corated_list:
            numerator += (bidx_dict1[corated] - avg1)*(bidx_dict2[corated] - avg2)
            denumerator1 += (bidx_dict1[corated] - avg1)**2 
            denumerator2 += (bidx_dict2[corated] - avg2)**2 
        if denumerator1 == 0 or denumerator2 == 0:
            pearson = 0
        else:
            pearson = numerator / (math.sqrt(denumerator1) * math.sqrt(denumerator2))
    else:
        pearson = -1
    
    return (uidx1, uidx2, pearson)

if cf_type == 'item_based':
    bus_user_star_rdd = jsonLine_rdd.map(lambda jLine:(jLine['business_id'],(jLine['user_id'], jLine['stars'])))                        .map(lambda bigPair:(bid_index_dict[bigPair[0]],(uid_index_dict[bigPair[1][0]], bigPair[1][1])))                        .groupByKey().mapValues(lambda user_star_pair_group:list(user_star_pair_group))                        .map(lambda busPair:processSameBusSameUserMultiStars(busPair))                        .filter(lambda bigPair:len(bigPair[1])>=3)
                          
    bidx_uidx_star_dict = dict(bus_user_star_rdd.collect())
    qualified_bidx = list(bidx_uidx_star_dict.keys())
    pairwise_bidx = list(itertools.combinations(qualified_bidx, 2))
    
    candidates_pairs_rdd = sc.parallelize(pairwise_bidx, 12).map(lambda pair: findCandidatesPair(pair))                         .filter(lambda bigPair:bigPair[1]>=3).map(lambda bigPair:(bigPair[0][0], bigPair[0][1]))    
    pearson = candidates_pairs_rdd.map(lambda pair:pearsonCorrelation(pair))                              .filter(lambda resultTuple:resultTuple[2]>0).collect()
    
    bindex_id_dict = bid_index_rdd.map(lambda pair:(pair[1], pair[0])).collectAsMap()
    modelWriter = open(model_file_path, 'w')
    for pair in pearson:
        b1_id = bindex_id_dict[pair[0]]
        b2_id = bindex_id_dict[pair[1]]
        jsonPair = {'b1':b1_id, 'b2':b2_id, 'sim':pair[2]}
        modelWriter.write(json.dumps(jsonPair) + '\n')
    modelWriter.close()
else:
    # preparing for minhash
    uidx_bidx_rdd = jsonLine_rdd.map(lambda jsonLine:(jsonLine['user_id'], jsonLine['business_id']))                               .groupByKey().mapValues(lambda busList:list(set(busList)))                               .map(lambda pair: (uid_index_dict[pair[0]], [bid_index_dict[bid] for bid in pair[1]]))           
    num_hash_func = 50
    hash_func_list = list()
    for _ in range(num_hash_func):
        param_a = random.randint(1,19980511)
        param_b = random.randint(1,19971218)
        param_p = 21842137 # a random 8-digit prime number
        hash_func_list.append((param_a, param_b, param_p))
        
    minhash_signature_matrix = uidx_bidx_rdd.map(lambda pair:minhash(pair, len(bid_index_dict)))
    
    num_bands = 50
    lsh_result_rdd = minhash_signature_matrix.map(lambda sig:createBands(sig, num_bands))                                         .flatMap(lambda x:x).groupByKey()                                         .mapValues(lambda t:list(t)).filter(lambda pair:len(pair[1])>1)                                         .flatMap(lambda pair: list(itertools.combinations(pair[1], 2)))                                         .distinct()
    char_matrix = dict(uidx_bidx_rdd.collect())
    jaccard_result_rdd = lsh_result_rdd.map(lambda candPair:computeJaccard(candPair)).filter(lambda pair:pair[2]>=0.01)

    uidx_bidx_star_dict = jsonLine_rdd.map(lambda jLine:(jLine['user_id'],(jLine['business_id'], jLine['stars'])))                        .map(lambda bigPair:(uid_index_dict[bigPair[0]],(bid_index_dict[bigPair[1][0]], bigPair[1][1])))                        .groupByKey().mapValues(lambda bus_star_pair_group:list(bus_star_pair_group))                        .map(lambda pair:processSameUserSameBusMultiStars(pair))                        .filter(lambda pair:len(pair[1])>=3).collectAsMap()
    
    pearson_result = jaccard_result_rdd.map(lambda pair:computePearson(pair))                                       .filter(lambda bigPair:0 < bigPair[2]).collect()
    
    uindex_id_dict = uid_index_rdd.map(lambda pair:(pair[1], pair[0])).collectAsMap()
    modelWriter = open(model_file_path, 'w')
    for pair in pearson_result:
        u1_id = uindex_id_dict[pair[0]]
        u2_id = uindex_id_dict[pair[1]]
        jsonPair = {'u1':u1_id, 'u2':u2_id, 'sim':pair[2]}
        modelWriter.write(json.dumps(jsonPair) + '\n')
    modelWriter.close()


print('Time:', str(time.time()-startTime))

