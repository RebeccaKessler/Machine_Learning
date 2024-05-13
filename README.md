# Machine_Learning


## Welcome to the final report of the Group Swatch 

The objective of this project is to build a model that accurately predicts the difficulty level of French sentences.
This report is divided in five sections.
- [Team Approach and Task Division](#team-Approach-and-Task-Division)
- [Comparison of Different Models](#Comparison-of-Different-Models)
- [Final Model](#Final-Model)
- [Application of the Model](#Application-of-the-Model)
- [Final Words](#Final-Words)


# Team Approach and Task Division

We approached the task as a reiterative process, incrementally improving our models to predict the difficulty level of French sentences as accurately as possible. We started off with simple ML models inlcuding Logistic Regression, KNN, Decision Tree, and Random Forest before then advancing to more sophisticated large language models (LLMs) such as Bert and CamemBert. Within the different models, we used hyper-parameter optimization to find the best solution. Additionally, we generated a larger dataset using ChatGTP to be able to fine-tine our final model even better. 
Once we were satisifed with the performance of our final model, we started thinking about potential applications of our final model, an attractive user interfence, and an engaging concept for the video presentation. 
To succeed in this undertaking, We distirbuted the tasks among the members as follows:
- Giulia: creating the different simple ML models to predict the language level of French texts
- Rebecca: working on the advanced techniques and developing the user interface on streamlit.


# Comparison of Different Models


| Metric    | Logistic Regression | KNN | Decision Tree | Random Forest | Bert     | CamemBert  | 
|-----------|---------------------|-----|---------------|---------------|----------|------------|
| Precision |                     |     |               |               |          |            |              
| Recall    |                     |     |               |               |          |            |              
| F1-score  |                     |     |               |               |          |            |
| Accuracy  |                     |     |               |               |          |    59%        |






# Final Model
Our final model is based on CamemBert. CamemBert is a large language model that was pretrained on a large corpus of French texts. It is based on RoBERT (Robustly Optimized BERT) which is an optimized version of the original BERT model. The CamemBert base model consists of 12 layers, 12 attention heads, 768 hidden size and a total paramterers of 110 million. 
To set up our final model, we went through the following steps:

- **Step 1:  Data collection:** 
Given the provided labelled training data, no data collection or cleaning is required per se. However, we used ChatGPT to generate additional datapoints which proved quite valuable in increasing the performance of the model in the end. To do so, we uploaded the provided training data set to ChatGPT and asked him/her to randomly generate additional French sentences and their difficulty level. The instructions to ChatGPT are crucial. We explicitly asked ChatGPT to generate sentences based on the provided training dataset to avoid it from generating completely different sentences. Moreover, we highlighted that the generation should be random to avoid hidden patterns in the sentences that could then lead to an overfitting issue later on.
- **Step 2: Data preprocessing & Tokenization:**
To feed our data to the CamemBert model, we needed to apply the Label Encoder to the column "difficulty" before tokenizing the data using the CamemBert tokenizer. The CamemBert tokenizer converts raw text into a numerical format that can be processed by the model.
- **Step 3: Load model:**
The CamemBert model is loaded with its pretrained weights. In our case, we load the CamembertForSequencesClassification which adds an additional linear classification layer to the pretrained model and randomizes the initial weigths.
- **Step 4: Define Training Parameters:**
The Camembert model allows to specifc various training arguments such as batch size, learning rate, training epochs, or weight decay. The combination of these parameters can signifcantly impact both the computational resources required to run the model as well as the performance (see step 7). The training parameters automatically include the AdamW optimizer to optimize the weights of the model and a learning rate scheduler which adjusts the learning rate as the model is fine-tuned. The default loss function is cross-entropy loss.
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
Finally, we can use the model to make predictions on the unlabelled test data. For this, we first re-train the model with the optimized parameters on the extended dataset (the one we generated with ChatGTP) and then use this model to make the final predictions. This provided the highest accuracy on the unlabelled data (+60%). Re-training on the original training set and then conducting the predictions provided slightly lower accuracy (however, not sigfnicantly lower (58.5%)). This difference can be explained by the fact that a larger dataset allows the model to learn better. 

**Some comments on the result**: The average accuracy of our CamemBert model is eqaul to 59% which is substantially better than the simple ML models. However, due to the complexity of the model, the accuracy shows some variation from iteration to iteration despite applying cross-validation. This is due to the randomness in the training process. Variations in model initialization and batch shuffling can affect the final accuracy of the model. The same applies when we retrain the model on the whole dataset and then make the predictions on the unlabelled datasets. 

Find our final model here:



# Application of the Model

For the application of our model, we decided to create a Streamlit app that can predict the French difficulty level of books. This tool may be helpful for language teachers who are looking for books that match the language level of their students as well as for people learning French and want to read a book that matches their skills. There is nothing more frustrating than reading a book that is way too difficult. Our app BOOKLY is simple and user-friendly. Upon loading the app, users can find the input field in the sidebar on the left. To predict the diffictulty level of a book, all they have to do is enter the title of the book and upload an excerpt of the book in docx format. Our app will then input this text into the ML model and output the difficulty level of the text. 
Each inputed text will be automatically saved to the app's library. When clicking on the library button in the sidebar on the left, users can brows through all previously uploaded text excerpts. The search can be filtered by title or language level. In this way, users can either check for the difficulty level of specific books without having to upload them themselves or search for books that match their language level. Consequently, the value of the app grows over time as more and more people upload excerpts of different books to the app, enlarging the library.

Explore our app: https://machinelearning-bookly.streamlit.app/

# Final Words

Watch our video here: 
