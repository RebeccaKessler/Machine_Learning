# Machine_Learning


## Welcome to the final report of the Group Swatch 

The objective of this project is to build a model that accurately predicts the difficulty level of French sentences.
This report is divided into six sections.
- [Team Approach and Task Division](#team-Approach-and-Task-Division)
- [Overview of Git Repository](#overview-of-git-repository)
- [Comparison of Different Models](#Comparison-of-Different-Models)
- [Other Techniques](#Other-Techniques)
- [Application of the Model](#Application-of-the-Model)
- [Final Words](#Final-Words)


# Team Approach and Task Division

We approached the task as a reiterative process, incrementally improving our models to predict the difficulty level of French sentences as accurately as possible. We started off with simple machine learning models inlcuding Logistic Regression, KNN, Decision Tree, and Random Forest before then advancing to more sophisticated large language models (LLMs) such as Distilbert, CamemBert, and Flaubert. Within the different models, we used hyper-parameter optimization to find the best solution. Additionally, we generated a larger dataset using ChatGTP to be able to fine-tine our final model even better. 
Once we were satisifed with the performance of our models, we started thinking about potential applications of our final model, an attractive user interfence, and an engaging concept for the video presentation. 
To succeed in this undertaking, we distributed the tasks among the members as follows:
- Giulia: creating the different basic machine learning models to predict the language level of French texts.
- Rebecca: working on the advanced techniques and developing the user interface on streamlit.

# Overview of Git Repository

In the provided git, the following folders and datasets can be found:
- Codes: python files containing the code for the basic models and the various Bert models.
- Streamlit: all the files required to build our user interface on streamlit
- Book examples: example book excerpts to try out our application
- Training dataset, unlabelled test data, ChatGPT generated training dataset (called "combined_random_french_sentences.csv), and the final submission


# Comparison of Different Models

| Metric                   | Logistic Regression | KNN | Decision Tree | Random Forest | Distilbert | CamemBert | Flaubert |
|--------------------------|---------------------|-----|---------------|---------------|------------|-----------|-------------|
| Precision                | 44.3%               | 42.9%| 29.9%         | 31%           | 55.5%      | 61.2%     | 60.4%       |
| Recall                   | 44.9%               | 37.6%| 29.8%         | 32%           | 55.0%      | 58.9%     | 59.2%       |
| F1-score                 | 44.2%               | 36.9%| 24.5%         | 29.8%         | 55.0%      | 59%       | 59.4%       |
| Highest Accuracy on Training Data| 45.1%               | 37.9%| 29.6%         | 32.2%         | 55.1%      | 58.9%     | 59.5%       |
| Higest Accuracy on Unlabeled Data| 46.5%               | 33.8%| 33.4%         | 38.9%         | 56.1%      | 60.2%     | 59.9 %       |

# Basic Models
Before moving to more sophisitcated models, we tried to predict the difficulty level of French sentences with the following models : Logistic Regression, k-Nearest Neighbors (KNN), Decision Tree, and Random Forest. For each of these models we predicted the accuracy on the training data as well as on the test data.

**Training and Testing Data**

First of all, we splitted the training dataset into training data and test data  with the train-test-split method. After the splitting, 80% of the dataset would be used to train the model, and the remaining 20% to test the model.
For each of the four models, we first trained the model on 80% of the data, and tested the model on 20% of the data. Then, we trained the model on the full training dataset, and tested the model on the unlabelled test data.

**Text Preprocessing**

The data we had needed to be preprocessed in order to be processed by our models. We used the TF-IDF vectorizer (Term Frequency-Inverse Document Frequency). This vectorizer converts text into numerical vectors, suitable for machine learning algorithms. Term Frequency measures the frequency of a term in a document, it indicates how often a certain word appears in a document relative to the total number of words in that document. The higher it is, the more that word appears in the document. Inverse Document Frequency measures the rarity of a term across all documents in a corpus. By multiplying TF with IDF, we obtain the importance of a term in a document relative to how often it appears across all documents. A high TF-IDF score means that a term is frequent in one document but rare across the corpus of documents, potentially bringing meaningful information. By applying the TF-IDF vectorizer to our data, each sentence is represented by a vector of terms with the corresponding TF-IDF value.

**Logistic Regression**

Out of the four basic models, the one performing the best is the logistic regression. Logistic Regression is a linear classification algorithm. It models the probability that a given input belongs to a particular class using a logistic function. The class predicted is the one that has the highest probability.

We started with a simple Logistic Regression, without specifying any parameters. We trained the model on 80% of the training data, tested it on 20% of the data, and got an accuracy of 45.1%. The precision, recall and F1 scores are all around 44%. This means that the model performance is quite balanced, it is able to identify similarly true positives (precision) and positives in general (recall), which is desirable in classification tasks such as this one.

By displaying the accuracy, precision, recall and F1 scores for each of the six individual classes, we noticed that the logistic regression predicts well the classes A1 and C2, with an accuracy of 64% and 62% respectively. However, for the remaining classes, the prediction is quite poor. This could be caused by an uneven distribution of the data. Consequently, if the training data disproportionally contains more A1 and C2 sentences, compared to sentences of other levels, it will impact the prediction accuracies since the model will be more trained on A1 and C2 sentences. In fact, after analysing the data, we find that there are more sentences of difficulty A1 (813) and C2 (807) in the training dataset, compared to sentences of other levels (795 on average). This could explain the difference in accuracies for different classes.

After the simple Logistic Regression, we performed hyperparameter tuning to find the optimal parameters for the model. Here's an overview of the parameters:

- **C** : This parameter controls the strength of regularization. A smaller value (ex.: C=0.01) leads to stronger regularization and a simpler model, a larger value (ex.: C=1.0) leads to weaker regularization and a more complex model. In other words, the higher it is the more the model will be complex at the cost of fitting the data.
- **Penalty** : This parameter specifies the type of regularization to apply. In our case, it's either ridge regularization, which helps prevent overfitting by penalizing the sum of the squared coefficients, or no regularization.
- **Solver** : This parameter specifies which algorithm to use for optimization.
- **Class weight** : It is used to handle imbalanced classes. It specifies if there is weight adjustment for the classes during training.
-  **Max iter** : It specifies the maximum number of iterations done by the solver before converging. Increasing the number of iterations should help the solver find a better solution.
-  **Tol** : This parameter sets the tolerance for stopping criteria. It determines when the algorithm should stop to iterate. Smaller values can lead to a more precise convergence, altough it takes more time to train.

We used GridSearchCV to go through the combinations of these parameters to find the best configuration for our logistic regression model. This method uses cross-validation to evaluate the performance of the model with each set of parameters.

Surprisingly, the optimal parameters that are expected to provide the best accuracy for the model (C: 10, Class weights: None, Max iter: 100, Penalty: 12, Solver: liblinear, Tol: 0.0001) actually result in the same accuracy as the previous logistic regression model, which had no specified parameters, namely 45.1%.

SAY WHY WE THINK THIS IS THE CASE

Finally, we retrained the model on the full training dataset without specifying any parameters, as previously noted, this would not improve the accuracy of prediction. After testing the model on the unlabelled test data, we obtain an accuracy of 46.5%.

**K-Nearest Neighbors (KNN)**

KNN is a non-parametric classification algorithm that assigns a class label to a data point, based on the majority class of its nearest neighbors. This model doesn't make assumptions about the underlying data distribution.

Similarly as for the Logistic Regression, we started with a K-Neighbors Classifier without specifying any parameter. After training the model on 80% of the training data, and testing it on 20% of the data, we got an accuracy of 31.9%. The precision, recall and F1 scores were lower than for the logistic regression model.

By examining the scores for the individual classes, we observed that sentences in the A1 class are correctly predicted with an accuracy of around 85%, which is significantly higher than the prediction accuracy for the other classes, lower than 25%.

To improve the KNN model's class prediction accuracy, we employed multiple loops to iterate over the model parameters and identify which are the optimal ones. The parameters we considered are the following: 

- **N Neighbors** : This parameter specifies the number of neighbors to use to make predictions. A larger number of neighbors makes the model less sensitive to noise in the data, but renders it more computationally resource-intensive.
- **P values** : When p=1, it corresponds to the Manhattan distance (L1 norm: distance calculated as the sum of the absolute differences of the coordinates), when p=2, it corresponds to the Euclidian distance (L2 norm: distance calculated as the square root of the sum of the squared differences of the coordinates). Euclidean distance is more sensitive to differences in magnitude, while Manhattan distance is more robust to outliers.
- **Weights** : This parameter specifies how to weight the contributions of the neighbors when making a prediction. When it's set to uniform, all neighbors are equally weighted, when it's set to distance, weights are inversely proportional to the distance, meaning that closer neighbors have a greater influence on the prediction.

We found that the best parameters are k=1 (1 neighbor), p=2 (Euclidian distance), and weight = distance. When training the model with these parameters on 80% of the data, and testing it on 20% of the data, we get an accuracy of 37.9%. We can consider it a great improvement.

Finally, to go a step further, we used GridSearchCV for hyperparameter tuning to identify the optimal model parameters. Not much differed from our previous approach using loops. We retained the same parameters for the p value and the weights in the parameter grid. The only change was allowing the number of neighbors to range from 1 to 10, to evaluate if this would enhance accuracy.

Unfortunately, this approach did not yield higher accuracy. In fact, it suggested different parameters, resulting in a lower accuracy of ADD THE ACCURACY, compared to the previous 37.9%. It is possible for hyperparameter tuning to fail in identifying the optimal parameters. Thus, it was beneficial to search for the best parameters using the loops.

As a final step, we retrained the KNN classifier on the full training dataset with the best parameters mentioned previously, and tested it on the unlabelled test data. We obtain an accuracy of 33.8%.

**Decision Tree**

A decision tree recursively splits the dataset into subsets based on the value of attributes. It's a tree-like structure where each internal node represents a feature, each branch represents a decision rule, and each leaf node represents the outcome or class label.

Once again, we started by using a Decision Tree Classifier without specifying any parameter. We trained the model on 80% of the training data and tested it on 20% of training data. We get an accuracy of 29.9%. The precision, recall and F1 scores are all around 29%, meaning that the model performance is balanced despite being poor. Similarly to the two previous models, the individual accuracy of the A1 class is higher compared to the accuracy of the other classes, altough the difference is minor this time. In fact, the A1 class accuracy is 55.4%, while for the other classes it ranges between 20-27%.

WHAT CAUSES THIS DIFFERENCE ???

Like for the KNN classifier, we looped over some model parameters to find the optimal parameters. These are the parameters we considered:

- **Criterion** : This parameter specifies which function to use to measure the quality of a split, it determines how the decision tree evaluates the potential splits at each node. If it's equal to Gini, it uses the Gini impurity to measure the quality of a split. If it's equal to Entropy, it uses the information gain based on entropy to measure it.
- **Max depth** : This parameter specifies the maximum depth of the tree, in other words, how many times the tree will split. Increasing the depth allows the tree to learn more in the data but also increases the risk of overfitting.

After looping over these model parameters, we found that the highest accuracy of 29.6% was obtained with the model parameters Criterion: entropy and Max depth: 5. However, this accuracy of 29.6% is lower than the initially obtained one, namely, 29.9%.

Once again, in order to find parameters yielding the highest accuracy, we used GridSearchCV to perform hyperparameter tuning. In the parameter grid, we retained the same parameters as for the loops, and additionally we allowed the max depth to range from 1 to 10. This time, the best parameters seemed to be Criterion: entropy and Max depth: 9. However, surprisingly, the accuracy obtained with these parameters is 29.6%. The same as with the previous best parameters and still lower than the initial one (29.9%).

Finally, we retrained the model on the full training dataset without specifying any parameter, then tested it on the unlabelled test data. We obtain an accuracy of 32.4%.

COMMENT ON IT !!!

**Random Forest**

The last basic model that we tried is the Random Forest Classifier. Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the mode of the classes of the individual trees. It improves upon the decision tree algorithm by reducing overfitting and increasing accuracy.

The Random Forest Classifier without any parameter specified, outputed an accuracy of 40.4%. Once again, the accuracy of the A1 class (77.7%) is disproportionately higher than the accuracy of the other classes (30% on average).

SAY WHYYYY ???

Like for the other models, we looped over model some model parameters to find the optimal ones. In this case, the parameters are the same as for the Decision Tree Classifier, with an additional one:

- **N estimators** : This parameter specifies the number of decision trees to be included in the random forest. Increasing the number of estimators generally improves the performance of the model by reducing overfitting. However, at the same time, it increases computational cost.

This method gives as best parameters N estimators: 11, Criterion: entropy, Max depth: 10 for an accuracy of 34.8%, not representing an improvement compared to the initial accuracy of 40.4%.

Then we performed hyperparameter tuning using GridSearchCV, which gave as optimal parameters N estimators: 9, Criterion: entropy, Max depth: 8. This, once again, doesn't seem to have worked as the accuracy obtained with those parameters is 29.6%, significantly lower than 40.4%.

Finally, we retrained the Random Forest Classifier on the full training data and tested it on the unlabelled test data. We obtain an accuracy of 38.9%.

# Other Techniques 
Our best performing models are based on CamemBert and Flaubert. Both of these models are large language models that were pretrained on French texts (CamemBert on the OSCAR corpus and Flaubert on a diverse French corpus, including sources such as Common Crawl, Wikipedia, and other text sources).  While CamemBert is based on RoBERT (Robustly Optimized BERT) which is an optimized version of the original BERT model, Flaubert is directly based on Bert. The CamemBert base model consists of 12 layers, 12 attention heads, 768 hidden size and a total paramterers of 110 million. Flaubert has the same amount of layers, attention heads, and hidden size but slightly more parameters.
To set up our final model for our task, we went through the following steps:

- **Step 1:  Data collection:** 
Given the provided labelled training data, no data collection or cleaning is required per se. However, we used ChatGPT to generate additional datapoints to see whether this would help us train our model better. To generate these sentencens, we uploaded the provided training data set to ChatGPT and asked him/her to randomly generate additional French sentences and their difficulty level. The instructions to ChatGPT are crucial. We explicitly asked ChatGPT to generate sentences based on the provided training dataset to avoid it from generating completely different sentences. Moreover, we highlighted that the generation should be random to avoid hidden patterns in the sentences that could then lead to an overfitting issue later on.
- **Step 2: Data preprocessing & Tokenization:**
To feed our data to the CamemBert or Flaubert model, we needed to apply the Label Encoder to the column "difficulty" before tokenizing the data using the CamemBert or Flaubert tokenizer, respectively. These tokenizers convert raw text into a numerical format that can be processed by the model.
- **Step 3: Load model:**
The model is loaded with its pretrained weights. In our case, we load the CamembertForSequencesClassification or FlaubertForSequencesClassification which adds an additional linear classification layer to the pretrained model and randomizes the initial weigths.
- **Step 4: Define Training Parameters:**
The models allow to specify various training arguments such as batch size, learning rate, training epochs, or weight decay. The combination of these parameters can signifcantly impact both the computational resources required to run the model as well as the performance (see step 7). The training parameters automatically include the AdamW optimizer to optimize the weights of the model and a learning rate scheduler which adjusts the learning rate as the model is fine-tuned. The default loss function is cross-entropy loss.
- **Step 5: Fine-tuning the model:**
Now it is time to fine-tune the model on our training data using the defined training parameters. Fine-tuning allows to adapt a pre-trained model to perform a specifc task, in our case to predict the difficulty of French sentences. Hence, fine-tuning allows the model to specialize in the required task. During the fine-tuning process, the model loops through several key steps in each epoch:
  - Forward pass: the inputs (tokzenized text) are passed through the model to get predictions.
  - Loss calculation: the loss between the predicted and true labels is calculated using cross-entropy.
  - Backward pass: the loss is backpropagated to calculate gradients.
  - Parameter update: gradients are used to then update model weigths via the optimizer AdamW.

We also implemented a K-fold cross-validation to obtain a more robust performance. K-fold cross validation means that the data is divided into "k" equal parts. Each part is used as a validation set once while the others serve as the training set (20/80 split), rotating through all "k" parts. In our case, we set k to 5 to not overwhelm our computational resources. 

- **Step 6: Evaluation:**
After each fold, the model is evalaute on the evaluation dataset. Here we primarily use accuracy as the evaluation metric. After the five folds, we calculate the final accuracy by taking the average over all folds.
- **Step 7: Optimization:**
To increase the performance (i.e. accuracy) of the model we set up a hyper-optimization process using optina. This helps us  find a good combination of parameters (epochs, batch size, and learning rate) which we then further adjust manually until we are satisfied with the performance level of the model.
- **Step 8: Prediction:**
Finally, we can use the model to make predictions on the unlabelled test data. For this, we first re-train the model with the optimized parameters on the full dataset and then use this model to make the final predictions. We then did the same using our extended dataset. This provided the highest accuracy on the unlabelled data. Re-training on the original training set and then conducting the predictions provided slightly lower accuracy (however, not sigfnicantly lower). This difference can be explained by the fact that a larger dataset allows the model to learn better. However, the results obtained by retraining on the extended dataset proved to be a lot less robust and would fluctuate a lot each time we run the model. In our final model (see next paragraph), we hence solely relied on the provided training data.

**Combination of Models**

After finalizing our individual models, we decided to go one step further and see what would happen if we were to combine the two models. Hence, we first fine-tune the CamemBert and Flaubert model individually on the training data and let them each make their predictions on the unlabelled data. These predictions are made in the form of a probability distribution over the six classes (A1, A2, ...). To combine the predictions of the two models, we can simply take the average of the two outputed probability distributions and use the resulting probabilities for the final prediction. This allowed us to further increase the accuracy of the predictions on the unlabelled data (61,8%). 


**Discussion of results**

The average training accuracy of both the CamemBert and Flaubert model is about  59% and the testing accuracy about 60% which is substantially better than the simple machine learning models. Hence, neural networks seem to be able to learn better from the provided training data and make correct predictions. Combining the two models allows to further increase this accuracy to above 61%. However, due to the complexity of these neural networks, the accuracy shows some variation each time we run the model despite applying cross-validation. This is due to the randomness in the training process. Variations in model initialization and batch shuffling can affect the final accuracy of the model. The same applies when we retrain the model on the whole dataset and then make the predictions on the unlabelled datasets. This, however, is not uncommon for neural networks. 

Moreover, both of the models seems to be experiencing some overfitting. While the training and validation loss initially both decrease, the validation loss starts to stagnate at some point and then increases again. We tried to counteract the overfitting my experimenting with various techniques such as dropout rates, learning rate scheduler, and weight decay. Surprisingly, however, most of these techniques led to lower accuracy on both the training and test data. While this can happen on the training data, we were at least expecting the accuracy on the test data to get better when controlling for overfitting as overfitting usually leads to poorer generalization of models. Hence, we were expecting that reducing the validation loss during the training would lead to a higher accuracy. However, this was not the case. This discrepancy can be due to several reasons. The increasing validation loss might only hint at the start of overfitting and might reflect a decreasing confidence in the correct predictions (i.e. probabilities). Consequently, the model might start to predict the correct class with lower probability. However, the predictions might still be correct, leading to the higher observed accuracy. If we were to prolong the training phase, we would expect the accuracy to decrease at some point. The regularization technique that proved to be the most effective in reducing the validation loss while not reducing the accuracy was as slight increase in the weight decay.


# Application of the Model

For the application of our model, we decided to create a Streamlit app that can predict the French difficulty level of books. This tool may be helpful for language teachers who are looking for books that match the language level of their students as well as for people learning French and want to read a book that matches their skills. There is nothing more frustrating than reading a book that is way too difficult. Our app BOOKLY is simple and user-friendly. Upon loading the app, users can find the input field in the sidebar on the left. To predict the diffictulty level of a book, all they have to do is enter the title of the book and upload an excerpt of the book in docx format. Our app will then input this text into the ML model and output the difficulty level of the text. 
Each inputed text will be automatically saved to the app's library. When clicking on the library button in the sidebar on the left, users can brows through all previously uploaded text excerpts. The search can be filtered by title or language level. In this way, users can either check for the difficulty level of specific books without having to upload them themselves or search for books that match their language level. Consequently, the value of the app grows over time as more and more people upload excerpts of different books to the app, enlarging the library.
(note: the app is running on the logisitc regression model as the Bert models were to large to upload to our github).

Explore our app: https://machinelearning-bookly.streamlit.app/

# Final Words

Watch our video here: 
