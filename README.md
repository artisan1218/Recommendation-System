# Recommendation-System (Content-based, Item-based Collaborative Filtering, User-based Collaborative Filtering)

Note: The Recommendation System will utilize the data from yelp.com
- train_review.json – the main file that contains the review data, RS will primarily be working with this file. 
- test_review.json – containing only the target user and business pairs for prediction tasks
- test_review_ratings.json – containing the ground truth rating for the testing pairs
- stopwords - containing common stopwords that will be used when calculating TFIDF score.

### The Recommendation System will be divided into three tasks, each uses different algorithm to accomplish the recommendation. 
* task1 will find similar business pairs in the train_review.json file. Algorithms used are: MinHash and Locality Sensitive Hashing, Jaccard Similarity
* task2 is the content-based RS which will generate profiles from review texts for users and businesses in the train_review.json file. Algorithms used are: Calculation of TF-IDF score and Cosine Similarity.
* task3 is the Collaborative Filtering Recommendation System which has two cases: Item-based CF and User-based CF.
  1. Item-based CF: the RS is built by computing the Pearson correlation for the business pairs with at least three co-rated users and use 3 or 5 neighbors who are most similar to targeted business.
  2. User-based CF: MinHash and LSH is used first to identify similar users to reduce the number of pairs needed to compute Pearson Correlation. After identifying the similar users based on their jaccard similarity, RS will compute the Pearson Correlation for all candidates user pairs and make the prediction. 

### Output Demo
* task1: ![image](https://user-images.githubusercontent.com/25105806/113206117-e9442500-9223-11eb-85f4-ce7f2cab3bbe.png)
  * b1 and b2 are the business id
  * sim is the jaccard similarity of b1 and b2
* task2: ![image](https://user-images.githubusercontent.com/25105806/113206164-fb25c800-9223-11eb-8894-3f8b347bb113.png)
  * user_id and business_id pair means 'if a user would prefer to review a business'
  * sim is the calculated(predicted) cosine distance between the profile vectors.
* task3 User-based CF Pearson Correlation Model: ![image](https://user-images.githubusercontent.com/25105806/113206248-1264b580-9224-11eb-933b-f13deef2045d.png) 
  * u1 and u2 are the user id
  * sim is the Pearson Correlation between these two users  
* task3 Item-based CF Pearson Correlation Model: ![image](https://user-images.githubusercontent.com/25105806/113206200-0547c680-9224-11eb-84dd-063e8a2413db.png)
  * b1 and b2 are the business id
  * sim is the Pearson Correlation between these two business  
* task3 prediction result: ![image](https://user-images.githubusercontent.com/25105806/113206379-3d4f0980-9224-11eb-8511-25fcadccf637.png)
  * user_id and business_id stands for 'this user will likely rate this business with this star'
  * stars is simply the predicted rating 

