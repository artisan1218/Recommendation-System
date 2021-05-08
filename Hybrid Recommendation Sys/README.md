# Hybrid Recommendation System

Note: The Recommendation System will utilize the data from yelp.com
- business_avg.json - containing the average rating for all businesses
- user_avg.json - containing the average rating for all users
- user.json - containing the user-related information for each user
- train_review.json – the main file that contains the review data, RS will primarily be working with this file
- test_review.json – containing only the target user and business pairs for prediction tasks
- test_review_ratings.json – containing the ground truth rating for the testing pairs
- The file is preprocessed first using Apache Spark

### Introduction

 1. My implementation of the recommendation system primarily uses Singular-Value-Decomposition (SVD), XGBoost, Up-sampling, User profile (Friends).
 2. Libraries and dependencies: XGBoost, Surprise, Sklearn
 3. The hybrid approaches I used include Mixed Hybrid and Feature Augmentation. 

### Training process:
Feature Augmentation is for the SVD model. Since we only have limited amount of training data, to better train the SVD model, I up-sampled the training set by randomly adding user-business-avg star pairs to current training set. Adding 300,000 pairs will yield a better result without overfitting the model.
For the XGBoost model, the training features are all from user.json, through the test, these features are most useful among all others:
`feature_names = ['uavg', 'bavg', 'useful', 'funny', 'cool', 'fans', 'elite', 'user_num', 'bus_num', 'rating']`, `uavg` is the average rating of this user, while `bavg` is simply the average rating this business ever received.\
Then I trained the rest of aforementioned models separately and save them to files, these files are:
 * bid2idx_dict_model.txt
 * uid2idx_dict_model.txt
 * friends_model.json
 * svd_model.model
 * xgb_model.json

They are all in the model folder.

### Prediction process:
In the prediction process, the primary model will be XGBoost, which is good enough for most of the testing pairs, which is seen/known user and seen/known business. But there are several cases that XGBoost cannot make a good prediction on:
 1. Case 1: Unseen business with seen user
    * Predict the ratings through the friends of the known user. If this user has enough friends, we can predict the rating by taking the average ratings of all the friends. This approach assumes that the user will have similar taste with his/her friends (Otherwise they might not be friends). If this user did not have any friends on records, we can simply predict the rating by using this user’s average rating over all businesses. 
 2.	Case 2: Unseen user with seen business
    *	Predict the ratings through the friends of the user. Because the absence of the ratings of a user does not imply the absence of his/her friends. We can still use similar way to predict rating as in case 1. If the user does not exist in friends model, then we can use the average rating of this business as the predicted rating because the average rating of a business, although might not reflect this unknown user’s rating accurately, is still representative of the business’s level and therefore generate roughly accurate prediction.
 3.	Case 3: Both user and business are unseen
    * This means we do not have any information on this user nor this business. This is a completely new pair. I predicted this by using weighted hybrid of average stars of all users and average stars of all business. Both will have a 0.5 weight. 
  
The SVD and XGBoost will both predict on the same dataset and generate two prediction result, then I did a weighted sum of these two prediction results. SVD result will have a weight of 0.15 and XGBoost result will have a weight of 0.85. This way, I combined different recommenders together to generate a result jointly (Mixed Hybrid).

### Final result
This project is ranked the third place at USC Data Mining (Recommendaation System) Competition 2021 with final score of 2709 and RMSE of 1.1498
   ![image](https://user-images.githubusercontent.com/25105806/117549836-77fd4c00-aff1-11eb-82a6-0cfe6b925cd7.png)

