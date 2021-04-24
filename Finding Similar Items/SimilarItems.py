from pyspark import SparkContext
import itertools
import json
import math
import random
import time
import sys


startTime = time.time()
input_file_path = sys.argv[1]#r'C:\Users\11921\Downloads\data\train_review.json'
output_file_path = sys.argv[2]#r'task1_output.json'

sc = SparkContext.getOrCreate()
user_bus_pair = sc.textFile(input_file_path).map(lambda line:json.loads(line))                        .map(lambda jsonLine:(jsonLine['user_id'],jsonLine['business_id']))



# To create table of userid/busid
uid_index = user_bus_pair.map(lambda pair:pair[0]).distinct().zipWithIndex().collect()
bid_index = user_bus_pair.map(lambda pair:pair[1]).distinct().zipWithIndex().collect()
#uid_index = user_bus_pair.map(lambda pair:pair[0]).distinct().sortBy(lambda uid:uid).zipWithIndex().collect()
#bid_index = user_bus_pair.map(lambda pair:pair[1]).distinct().sortBy(lambda bid:bid).zipWithIndex().collect()

uid_uindex_dict = {id_index_pair[0]:id_index_pair[1] for id_index_pair in uid_index}
bid_bindex_dict = {id_index_pair[0]:id_index_pair[1] for id_index_pair in bid_index}



# The original_rating_table is of the format (bus:[user1,user6,...]): 
# [        
#  (--9e1ONYQuAa-CB_Rrw7Tw, [user1, user4, user887, user4638, ...]),
#  ......
# ]
# It will be converted to below format.
# The bidx_uidx_table is of the format (bidx:[uidx1,uidx4,...]): 
# [   
#  (0, [0, 50, 227, 311, ...]),     
#  (1, [1, 4, 887, 4638, ...]),
#  ......
# ]
# each row(list) of bidx_uidx_table is used to compute minhash signature


bidx_uidx_rdd = sc.textFile(input_file_path).map(lambda line:json.loads(line))                            .map(lambda jsonLine:(jsonLine['business_id'],jsonLine['user_id']))                            .groupByKey().mapValues(lambda userList: list(set(userList)))                            .map(lambda pair:(bid_bindex_dict[pair[0]], [uid_uindex_dict[userId] for userId in pair[1]]))                            .sortBy(lambda pair:pair[0]).map(lambda pair:(pair[0], sorted(pair[1])))


def minhash(row, num_bins):
    bidx = row[0]
    uidx_list = row[1]
    minhash_signature_list = list()
    for params in hash_func_list: # generate 40 different hash function
        hashed_uidx_list = list() 
        param_a = params[0]
        param_b = params[1]
        param_p = params[2]
        param_m = num_bins
        for uidx in uidx_list: # hash all the uidx using each hash function and find minhash signature for that hash func
            hashed_uidx = ((param_a*int(uidx)+param_b)%param_p)%param_m
            hashed_uidx_list.append(hashed_uidx)
        minhash_signature = min(hashed_uidx_list) # 1 of the 40 minhash signature for this business
        minhash_signature_list.append(minhash_signature)
    return (bidx, minhash_signature_list)


# minhash_signature_matrix is list of list of format below
# each list in it represent 40 minhash signature of that business
# [
#  bus0: (0, [ 75, 70,134,  39,37]),
#  bus1: (1, [979,103,410,1198,51]),
#  ...
# ]
num_hash_func = 40

hash_func_list = list()
for _ in range(num_hash_func):
    param_a = random.randint(1,19980511)
    param_b = random.randint(1,19971218)
    param_p = 21842137 # a random 8-digit prime number
    hash_func_list.append((param_a, param_b, param_p))
    
minhash_signature_matrix = bidx_uidx_rdd.map(lambda row:minhash(row, len(uid_uindex_dict)))    


# bus_sig is tuple of (bidx, [sig1, sig2, ...])
def createBands(bus_sig, band_num):
    bidx = bus_sig[0]
    sig_list = bus_sig[1]
    bus_band = list()
    row_num = int(num_hash_func/band_num) # in this case: 40/10 = 4
    for i in range(band_num):
        band_tag = 'band'+str(i)
        bus_band.append((band_tag, (bidx, sig_list[i * row_num: i * row_num + row_num])))
    return bus_band

def lsh(bandTuple):
    band_tag = bandTuple[0]
    band_list = bandTuple[1]
    band_lsh_result = list()
    for bandTuple in band_list:
        bidx = bandTuple[0]
        hash_sig = bandTuple[1]
        lsh_sig = str(hash(tuple(hash_sig)))
        band_lsh_result.append((lsh_sig, bidx))
        
    return band_lsh_result

num_bands = 40
lsh_result = minhash_signature_matrix.map(lambda bus_sig:createBands(bus_sig, num_bands)).flatMap(lambda x:x)                              .groupByKey().mapValues(lambda t:list(t)).sortBy(lambda t:t[0])                              .map(lambda bandTuple:lsh(bandTuple)).collect()



def findCandidates(band_list):
    candidateList = list()
    candidatePair_result = list()
    for band in band_list:
        tmp_dict = dict()
        for twoTuple in band:
            if twoTuple[0] not in tmp_dict.keys():
                tmp_dict[twoTuple[0]] = [twoTuple[1]]
            else:
                tmp_dict[twoTuple[0]].append(twoTuple[1])
        tmp_list = [hashList for hashList in tmp_dict.values() if len(hashList)>1]
        for l in tmp_list:
            candidateList.append(tuple(l))
    for bidx_tuple in candidateList:
        tmp_list = list(itertools.combinations(list(bidx_tuple), 2))
        for pair in tmp_list:
            candidatePair_result.append(pair)
    return candidatePair_result

def verify(can_tuple):
    can1 = can_tuple[0]
    can2 = can_tuple[1]
    userList1 = char_matrix[can1][1] # char_matrix is sorted, so busIdx is equal to its index in char_matrix
    userList2 = char_matrix[can2][1]
    jaccard = len(set(userList1).intersection(set(userList2)))/len(set(userList1).union(set(userList2)))
    return (can1, can2, jaccard)


candidate_bidx_pair = findCandidates(lsh_result)
char_matrix = bidx_uidx_rdd.collect()    
final_output = sc.parallelize(candidate_bidx_pair).distinct().map(lambda can_tuple:verify(can_tuple))                 .filter(lambda threeTuple: threeTuple[2]>=0.05).collect()


bid_list = list(bid_bindex_dict.keys())
bidx_list = list(bid_bindex_dict.values())
id_final_output = list()
for threeTuple in final_output:
    can1_idx = threeTuple[0]
    can2_idx = threeTuple[1]
    jac_sim = threeTuple[2]
    cand1_id = bidx_list.index(can1_idx)
    cand2_id = bidx_list.index(can2_idx)
    id_final_output.append((bid_list[cand1_id],bid_list[cand2_id],jac_sim))

outputFile = open(output_file_path, 'w')
for threeTuple in id_final_output:
    jsonLine = {"b1": threeTuple[0], "b2": threeTuple[1], "sim": threeTuple[2]}
    outputFile.write(json.dumps(jsonLine) + '\n')
outputFile.close()

print('time: ' + str(time.time()-startTime))





