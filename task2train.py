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
output_file_path = sys.argv[2]#r'task2.model'
stopwords_path = sys.argv[3]#r'C:\Users\11921\Downloads\data\stopwords'

stopwordReader = open(stopwords_path,'r')
stopwords = set([word[:-1] for word in stopwordReader.readlines()])
stopwordReader.close()

def preprocess(review):
    excludedChars = string.punctuation+string.digits
    #puncLess_numLess_review = review.translate(str.maketrans(excludedChars, ' '*len(excludedChars)))
    puncLess_numLess_review = review.translate(str.maketrans('','',excludedChars))
    clean_review_list = [reviewWord for reviewWord in puncLess_numLess_review.lower().split() if reviewWord not in stopwords]
    clean_review = ' '.join(clean_review_list)
    return clean_review

conf = SparkConf().setMaster("local")         .set("spark.executor.memory", "4g")         .set("spark.driver.memory", "4g")

# bus_reviews is a list of tuples. Each tuple is a pair of businessID and clean review text str
sc = SparkContext(conf=conf).getOrCreate()
bus_reviews_rdd = sc.textFile(input_file_path).map(lambda line:json.loads(line))                        .map(lambda jsonLine:(jsonLine['business_id'],jsonLine['text']))                        .reduceByKey(lambda review1, review2: str(review1) + ' ' + str(review2))                        .map(lambda pair:(pair[0], preprocess(pair[1])))
bus_reviews = bus_reviews_rdd.collect()


# In[3]:


# Obtain the non-rare words list / remove the extremely rare words
# result_word_set is a set of non-rare words. Only words in this set can be used to calculate TFIDF
word_cnt_list = list()
for br_tuple in bus_reviews:
    reviewText = br_tuple[1]
    word_count_dict = Counter(reviewText.split())
    word_cnt_list.append(list(word_count_dict.items()))

word_cnt_dict = dict()
word_freq_dict = dict()
result_word = list()
total_num_words = 0
for word_cnt in word_cnt_list:
    for word in word_cnt:
        total_num_words+=1
        if word[0] not in word_cnt_dict.keys():
            word_cnt_dict[word[0]] = word[1]
        else:
            word_cnt_dict[word[0]] += word[1]
for word in word_cnt_dict.keys():
    word_freq_dict[word] = word_cnt_dict[word]/total_num_words

for pair in word_freq_dict.items():
    if pair[1] > 0.000001:
        result_word.append(pair[0])

result_word_set = set(result_word)


# In[4]:


# Remove the non-rare words from the bus_reviews_rdd / Keep only the non-rare words
def getNonRareWordList(bus_review):
    resultList = list()
    originalWordList = bus_review[1].split()
    for originalWord in originalWordList:
        if originalWord in result_word_set:
            resultList.append(originalWord)
    return (bus_review[0], resultList)

# review_list_rdd is tahe list of tuple. Each tuple is a pair of (busID, a list of review words without rare words)
review_list_rdd = bus_reviews_rdd.map(lambda bus_review:getNonRareWordList(bus_review))
#doc_reviewList = review_list_rdd.collect()


# In[5]:


# word_IDF_dict is a dict of unique words in all reviews and their corresponding IDF score
num_docs = review_list_rdd.count()   
word_IDF = review_list_rdd.map(lambda bid_wordList_tuple: [(word, 1) for word in set(bid_wordList_tuple[1])])                         .flatMap(lambda x:x).reduceByKey(lambda occu1,occu2:occu1+occu2)                         .map(lambda twoTuple:(twoTuple[0], math.log(num_docs/twoTuple[1],2))).collect()
word_IDF_dict = dict(word_IDF)


# In[6]:


def calculateTFIDF(twoTuple):
    wordList = twoTuple[1]
    word_count = dict(Counter(wordList))
    max_occu = max(word_count.values())
    result = list()
    for word_cnt_tuple in word_count.items():
        word = word_cnt_tuple[0]
        cnt = word_cnt_tuple[1]
        TF = int(cnt)/max_occu
        TFIDF = TF * word_IDF_dict[word]
        result.append((word, TFIDF))
    return (twoTuple[0], sorted(result, key=lambda t:-t[1])[:200])

#TFIDF is a list of tuple. Each tuple is a pair of (busID, a list of tuple of (unique word, corresponding TFIDF score))
TFIDF_rdd = review_list_rdd.map(lambda twoTuple:calculateTFIDF(twoTuple))


# In[7]:


word_index_dict = dict(TFIDF_rdd.map(lambda bus_words: [word_TFIDF[0] for word_TFIDF in bus_words[1]])                           .flatMap(lambda l:l).distinct().zipWithIndex().collect())


bus_profile = TFIDF_rdd.mapValues(lambda wordList:[word_index_dict[wordTuple[0]] for wordTuple in wordList]).collect()
bus_profile_dict = dict(bus_profile)


# In[10]:


user_profile = sc.textFile(input_file_path).map(lambda line:json.loads(line))                 .map(lambda jsonLine:(jsonLine['user_id'],jsonLine['business_id']))                 .groupByKey().mapValues(lambda x:list(set(x)))                 .mapValues(lambda busList:[bus_profile_dict[bus] for bus in busList])                 .mapValues(lambda bigL:[v for smallL in bigL for v in smallL])                 .mapValues(lambda l: list(set(l))).collect()
user_profile_dict = dict(user_profile)


# In[11]:


modelFile = open(output_file_path,'w')
user_profile_list = list(user_profile_dict.items())
for user in user_profile_list:
    modelFile.write(json.dumps( {'user': (user[0],user[1])} )+'\n')

lineNum = 0
bus_profile_list = list(bus_profile_dict.items())
for bus in bus_profile_list:
    if lineNum == len(bus_profile_list) - 1:
        modelFile.write(json.dumps( {'business': (bus[0],bus[1])} ))
    else:
        modelFile.write(json.dumps( {'business': (bus[0],bus[1])} )+'\n')
    lineNum+=1
    
modelFile.close()


# In[12]:


print('Duration:', str(time.time()-startTime))


# In[ ]:




