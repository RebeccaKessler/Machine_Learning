# Machine_Learning


## Welcome to the final report of the Group Swatch 

The objective of this project is to build a model that accurately predicts the difficulty level of French sentences.
This report is divided into five sections.
- [Team Approach and Task Division](#team-Approach-and-Task-Division)
- [Comparison of Different Models](#Comparison-of-Different-Models)
- [Final Model](#Final-Model)
- [Application of the Model](#Application-of-the-Model)
- [Final Words](#Final-Words)


# Team Approach and Task Division

We approached the task as a reiterative process, incrementally improving our models to predict the difficulty level of French sentences as accurately as possible. We started off with simple machine learning models inlcuding Logistic Regression, KNN, Decision Tree, and Random Forest before then advancing to more sophisticated large language models (LLMs) such as Distilbert, CamemBert, and Flaubert. Within the different models, we used hyper-parameter optimization to find the best solution. Additionally, we generated a larger dataset using ChatGTP to be able to fine-tine our final model even better. 
Once we were satisifed with the performance of our models, we started thinking about potential applications of our final model, an attractive user interfence, and an engaging concept for the video presentation. 
To succeed in this undertaking, we distributed the tasks among the members as follows:
- Giulia: creating the different simple machine learning models to predict the language level of French texts.
- Rebecca: working on the advanced techniques and developing the user interface on streamlit.


# Comparison of Different Models

| Metric                   | Logistic Regression | KNN | Decision Tree | Random Forest | Distilbert | CamemBert | Flaubert |
|--------------------------|---------------------|-----|---------------|---------------|------------|-----------|-------------|
| Precision                | 44.3%               | 42.9%| 28.9%         | 31%           | 55.7%      | 61.2%     | 60.4%       |
| Recall                   | 44.9%               | 37.6%| 29.8%         | 32%           | 55.4%      | 58.9%     | 59.2%       |
| F1-score                 | 44.2%               | 36.9%| 24.5%         | 29.8%         | 55.1%      | 59%       | 59.4%       |
| Highest Accuracy on Training Data| 45.1%               | 37.9%| 29.6%         | 32.2%         | 55.5%      | 58.9%     | 59.5%       |
| Higest Accuracy on Unlabeled Data| 46.5%               | 33.8%| 27.3%         | 25.2%         | 56.1%      | 60.2%     | 59.9 %       |



# Final Model
Our best performing models are based on CamemBert and Flaubert. Both models are large language model that were pretrained on French texts (CamemBert on the OSCAR corpus and Flaubert on a diverse French corpus, including sources such as Common Crawl, Wikipedia, and other text sources).  While CamemBert is based on RoBERT (Robustly Optimized BERT) which is an optimized version of the original BERT model, Flaubert is directly based on Bert. The CamemBert base model consists of 12 layers, 12 attention heads, 768 hidden size and a total paramterers of 110 million. Flaubert has the same amount of layers, attention heads, and hidden size but slightly more parameters.
To set up the models for our task, we went through the following steps:

- **Step 1:  Data collection:** 
Given the provided labelled training data, no data collection or cleaning is required per se. However, we used ChatGPT to generate additional datapoints which proved somewhat valuable in increasing the performance of the model in the end. To do so, we uploaded the provided training data set to ChatGPT and asked him/her to randomly generate additional French sentences and their difficulty level. The instructions to ChatGPT are crucial. We explicitly asked ChatGPT to generate sentences based on the provided training dataset to avoid it from generating completely different sentences. Moreover, we highlighted that the generation should be random to avoid hidden patterns in the sentences that could then lead to an overfitting issue later on.
- **Step 2: Data preprocessing & Tokenization:**
To feed our data to the CamemBert or Flaubert model, we needed to apply the Label Encoder to the column "difficulty" before tokenizing the data using the CamemBert or Flaubert tokenizer, respectively. These tokenizers convert raw text into a numerical format that can be processed by the model.
- **Step 3: Load model:**
The model is loaded with its pretrained weights. In our case, we load the CamembertForSequencesClassification or FlaubertForSequencesClassification which adds an additional linear classification layer to the pretrained model and randomizes the initial weigths.
- **Step 4: Define Training Parameters:**
The models allow to specify various training arguments such as batch size, learning rate, training epochs, or weight decay. The combination of these parameters can signifcantly impact both the computational resources required to run the model as well as the performance (see step 7). The training parameters automatically include the AdamW optimizer to optimize the weights of the model and a learning rate scheduler which adjusts the learning rate as the model is fine-tuned. The default loss function is cross-entropy loss.
- **Step 5: Fine-tuning the model:**
Now it is time to fine-tuned the model on our training data using the defined training parameters. Fine-tuning allows to adapt a pre-trained model to perform a specifc taks, in our case to predict the difficulty of French sentences. Hence, fine-tuning allows the model to specialize in the required task. During the fine-tuning process, the model loops through several key steps in each epoch:
  - Forward pass: the inputs (tokzenized text) are passed through the model to get predictions.
  - Loss calculation: the loss between the predicted and true labels is calculated using cross-entropy.
  - Backward pass: the loss is backpropagated to calculate gradients.
  - Parameter update: gradients are used to then update model weigths via the optimizer AdamW.

We also implemented a K-fold cross-validation to obtain a more robust performance. K-fold cross validation means that the data is divided into "k" equal parts. Each part is used as a validation set once while the others serve as the training set (20/80 split), rotating through all "k" parts. In our case, we set k to 5 to not overwhelm our computational resources. 

- **Step 6: Evaluation:**
After each fold, the model is evalaute on the evalaution dataset. Here we primarily use accuracy as the evalution metrics. After all five folds, we calculate the final accuracy by taking the average over all folds.
- **Step 7: Optimization:**
To increase the performance (i.e. accuracy) of the model we set up a hyper-optimization process using optina. This helps us  find a good combination of parameters (epochs, batch size, and learning rate) which we then further adjust manually until we are satisfied with the performance level of the model.
- **Step 8: Prediction:**
Finally, we can use the model to make predictions on the unlabelled test data. For this, we first re-train the model with the optimized parameters on the full dataset and then use this model to make the final predictions. We then did the same using our extended dataset. This provided the highest accuracy on the unlabelled data (+60%). Re-training on the original training set and then conducting the predictions provided slightly lower accuracy (however, not sigfnicantly lower). This difference can be explained by the fact that a larger dataset allows the model to learn better.


**Combination of Models**

We then go one step further and combine the two models. Hence, we first fine-tune the CamemBert and Flaubert model individually and let them each make their predictions on the unlabelled data. These predictions are in the form of a probability distribution over the six classes (A1, A2, ...). To combine the predictions of the two models, we then take the average over the two to obtain a final prediction. This substanitally increases the robustness of the results.

**Discussion of results**

The average training accuracy of both the CamemBert and Flaubert model is equal to 59% and the testing accuracy about 60% which is substantially better than the simple ML models. Hence, neural networks seem to be able to better learn from the provided training data and make correct predictions. However, due to the complexity of these neural networks, the accuracy shows some variation each time we run the model despite applying cross-validation. This is due to the randomness in the training process. Variations in model initialization and batch shuffling can affect the final accuracy of the model. The same applies when we retrain the model on the whole dataset and then make the predictions on the unlabelled datasets. This, however, is not uncommon for neural networks. Combining the two models and applying a majority voting allows to substantially decrease this variance and improve the robustness of the performance.
Moreover, the models seems to be experiencing some overfitting. While the training and validation loss initially both decrease, the validation loss starts to stagnate at some point, even slightly increasing again. We tried to counteract this overfitting my experimenting wiht various techniques such as dropout rates, learning rate scheduler, and weight decay. Surprisingly, however, most of these techniques lead to lower accuracy on both the training and test data. While this is to be expected on the training data, we would assume the accuracy on the test data to get better when controlling for overfitting as overfitting usually leads to poorer generalization of models. Hence, we were expecting the validation loss and accuracy to move in the same direction. This discrepancy can be due to several reasons. The increasing in validation loss during the training might only hint at the start of overfitting and might reflect a decreasing confidence in the correct predictions (i.e. probabilities). Consequently, the model might start to predict the correct class with lower probability. However, the predictions might still be correct, leading to the higher observed accuracy. If we were to prolong the training phase, we would expect the accuracy to decrease at some point. The regularization technique that proved to be most effective in reducing the validation loss while not reducing the accuracy is to sligthly increase in the weight decay.




# Application of the Model

For the application of our model, we decided to create a Streamlit app that can predict the French difficulty level of books. This tool may be helpful for language teachers who are looking for books that match the language level of their students as well as for people learning French and want to read a book that matches their skills. There is nothing more frustrating than reading a book that is way too difficult. Our app BOOKLY is simple and user-friendly. Upon loading the app, users can find the input field in the sidebar on the left. To predict the diffictulty level of a book, all they have to do is enter the title of the book and upload an excerpt of the book in docx format. Our app will then input this text into the ML model and output the difficulty level of the text. 
Each inputed text will be automatically saved to the app's library. When clicking on the library button in the sidebar on the left, users can brows through all previously uploaded text excerpts. The search can be filtered by title or language level. In this way, users can either check for the difficulty level of specific books without having to upload them themselves or search for books that match their language level. Consequently, the value of the app grows over time as more and more people upload excerpts of different books to the app, enlarging the library.
(note: the app is running on the logisitc regression model as the Bert models were to large to upload to our github).

Explore our app: https://machinelearning-bookly.streamlit.app/

# Final Words

Watch our video here: 
