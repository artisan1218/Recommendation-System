from pyspark import SparkContext
import json
import math
import time
import sys


startTime = time.time()
input_file_path = sys.argv[1]#r'C:\Users\11921\Downloads\data\test_review.json'
model_path = sys.argv[2]#r'task2.model'
output_file_path = sys.argv[3]#r'task2.predict'
sc = SparkContext().getOrCreate()


# In[3]:


user_profile = sc.textFile(model_path).map(lambda line:json.loads(line))                                      .map(lambda pair:list(pair.items()))                                      .flatMap(lambda x:x).filter(lambda pair:pair[0]=='user')                                      .map(lambda pair:(pair[1][0], pair[1][1])).collect()

bus_profile = sc.textFile(model_path).map(lambda line:json.loads(line))                                      .map(lambda pair:list(pair.items()))                                      .flatMap(lambda x:x).filter(lambda pair:pair[0]=='business')                                      .map(lambda pair:(pair[1][0], pair[1][1])).collect()


# In[10]:


user_profile_dict = dict(user_profile)
bus_profile_dict = dict(bus_profile)


# In[21]:


def verify(pair):
    u_prof = user_profile_dict.get(pair[0], 0)
    b_prof = bus_profile_dict.get(pair[1], 0)
    if u_prof!=0 and b_prof!=0:
        cosine_sim = len(set(u_prof).intersection(set(b_prof)))/(math.sqrt(len(u_prof)) * math.sqrt(len(b_prof)))
    else:
        cosine_sim = 0
    return (pair[0], pair[1], cosine_sim)
    

prediction = sc.textFile(input_file_path).map(lambda line:json.loads(line))                                         .map(lambda jLine: (jLine['user_id'],jLine['business_id']))                                         .map(lambda pair:verify(pair))                                         .filter(lambda threeTuple:threeTuple[2]>0.01)                                         .collect()


# In[25]:


outputFile = open(output_file_path, 'w')
for line in prediction:
    jsonLine = json.dumps({'user_id':line[0], 'business_id':line[1], 'sim':line[2]})
    outputFile.write(jsonLine + '\n')
outputFile.close()

print('Duration:', str(time.time()-startTime))


# In[ ]:




