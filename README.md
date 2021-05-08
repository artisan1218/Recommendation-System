# Recommendation-Systems

Note: The Recommendation System will utilize the data from yelp.com
- train_review.json – the main file that contains the review data, RS will primarily be working with this file. 
- test_review.json – containing only the target user and business pairs for prediction tasks
- test_review_ratings.json – containing the ground truth rating for the testing pairs
- stopwords - containing common stopwords that will be used when calculating TFIDF score.
- The file is preprocessed first using Apache Spark

### The Recommendation System will be divided into four subfolders, each uses different algorithm to accomplish the recommendation. 
* Similar Items.py will find similar business pairs in the train_review.json file. Algorithms used are: MinHash and Locality Sensitive Hashing, Jaccard Similarity
* Content-based RS.py is the content-based RS which will generate profiles from review texts for users and businesses in the train_review.json file. Algorithms used are: Calculation of TF-IDF score and Cosine Similarity.
* CF is the Collaborative Filtering Recommendation System which has two cases: Item-based CF and User-based CF.
  1. Item-based CF: the RS is built by computing the Pearson correlation for the business pairs with at least three co-rated users and use 3 or 5 neighbors who are most similar to targeted business.
  2. User-based CF: MinHash and LSH is used first to identify similar users to reduce the number of pairs needed to compute Pearson Correlation. After identifying the similar users based on their jaccard similarity, RS will compute the Pearson Correlation for all candidates user pairs and make the prediction. 

### Output Demo
* Similar Items: ![image](https://user-images.githubusercontent.com/25105806/113206117-e9442500-9223-11eb-85f4-ce7f2cab3bbe.png)
  * b1 and b2 are the business id
  * sim is the jaccard similarity of b1 and b2
* Content-based RS: ![image](https://user-images.githubusercontent.com/25105806/113206164-fb25c800-9223-11eb-8894-3f8b347bb113.png)
  * user_id and business_id pair means 'if a user would prefer to review a business'
  * sim is the calculated(predicted) cosine distance between the profile vectors.
* User-based CF Pearson Correlation Model: ![image](https://user-images.githubusercontent.com/25105806/113206248-1264b580-9224-11eb-933b-f13deef2045d.png) 
  * u1 and u2 are the user id
  * sim is the Pearson Correlation between these two users  
* Item-based CF Pearson Correlation Model: ![image](https://user-images.githubusercontent.com/25105806/113206200-0547c680-9224-11eb-84dd-063e8a2413db.png)
  * b1 and b2 are the business id
  * sim is the Pearson Correlation between these two business  
* CF prediction result: ![image](https://user-images.githubusercontent.com/25105806/113206379-3d4f0980-9224-11eb-8511-25fcadccf637.png)
  * user_id and business_id stands for 'this user will likely rate this business with this star'
  * stars is simply the predicted rating 

### Model and prediction accuracy/precision/recall/RMSE
1. Similar business pairs 
   1. precision: 1.0
   2. recall: 0.9582400942205771
2. Content-based RS
   1. precision (test set): 1.0
   2. recall (test set): 0.999469477863536
3. CF model
   1. item-based CF model
      1. precision: 0.9641450981844213
      2. recall: 0.9805068470797926
   2. user-based CF model
      1. precision: 0.9573746593617223
      2. recall: 0.8276633759390503
4. CF prediction
   1. item-based RMSE (test set): 0.9023539405054186
   2. user-based RMSE (test set): 0.9901023647008427

### Algorithm and Mathematical inference of the model
1. Cosine Similarity: 

    <img src="https://user-images.githubusercontent.com/25105806/113209393-de8b8f00-9227-11eb-81be-64dc2cfe2ec4.png" width="50%" height="50%">
2. Normalized Term Frequency:
    
    <img src="https://user-images.githubusercontent.com/25105806/113209177-91a7b880-9227-11eb-88a6-9380099b4c58.png" width="50%" height="50%">
3. Inverse Document Frequency: 
  
    <img src="https://user-images.githubusercontent.com/25105806/113209263-ad12c380-9227-11eb-9bde-bb70acc556d1.png" width="50%" height="50%">
4. TF-IDF score: 
 
    <img src="https://user-images.githubusercontent.com/25105806/113209289-b734c200-9227-11eb-903e-46e77396c2d5.png" width="35%" height="35%">
5. Jaccard Similarity and distance: 

    <img src="https://user-images.githubusercontent.com/25105806/113209494-fd8a2100-9227-11eb-8c88-3a22446cb77b.png" width="40%" height="40%">
6. User-based CF Pearson Correlation: 
 
    <img src="https://user-images.githubusercontent.com/25105806/113209564-12ff4b00-9228-11eb-91d3-55b961c97346.png" width="45%" height="45%">
7. User-based CF prediction using Pearson Correlation: 

    <img src="https://user-images.githubusercontent.com/25105806/113209629-26aab180-9228-11eb-9e0a-52fe12d8a423.png" width="45%" height="45%">
8. Item-based CF Pearson Correlation: 

    <img src="https://user-images.githubusercontent.com/25105806/113209694-3aeeae80-9228-11eb-972d-36476fc96b50.png" width="45%" height="45%">
9. Item-based CF prediction using Pearson Correlation:

    <img src="https://user-images.githubusercontent.com/25105806/113209749-49d56100-9228-11eb-8605-b1c7e48c0891.png" width="30%" height="30%">






