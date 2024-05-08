# Machine_Learning


## Welcome to final report of the Group Swatch 

In this report, you can find the following sections:
- [Team Approach and Task Division](#team-Approach-and-Task-Division)
- [Comparison of Different Models](#Comparison-of-Different-Models)
- [Final Model](#Final-Model)
- [Application of the Model](#Application-of-the-Model)
- [Final Words](#Final-Words)


# Team Approach and Task Division

We approached the task as a reiterative process, incrementally improving our model to predict the language level of French texts. We started off with simple ML models inlcuding Logistic Regression, KNN, Decision Tree, and Random Forest before then advancing to more sophisticated large language models (LLMs) such as Bert and CamamBert. Within the different models, we used hyper-parameter optimization to find the best solution. Additionally, we generated a larger dataset using ChatGTP to be able to fine-tine our final model even better. 
Once we were satisifed with the performance of our final model, we started to think about potential applications of our final model, an attractive user interfence, and an engaging concept for the video presentation. 
To succeed in this undertaking, We distirbuted the tasks among the members as follows:
- Giulia: creating the different simple ML models to predict the language level of French texts
- Rebecca: workign on the advanced techniques and developing the user interface on streamlit.


# Comparison of Different Models


| Metric    | Logistic Regression | KNN | Decision Tree | Random Forest | Bert     | CamemBert  | 
|-----------|---------------------|-----|---------------|---------------|----------|------------|
| Precision |                     |     |               |               |          |            |              
| Recall    |                     |     |               |               |          |            |              
| F1-score  |                     |     |               |               |          |            |
| Accuracy  |                     |     |               |               |          |            |






# Final Model
Our final model is based on CamemBert. CamemBert is a large language model that was pretrained on a large corpus of French texts. It is based on RoBERT (Robustly Optimized BERT) which is an optimized version of the original BERT model. The CamemBert base model consists of 12 layers, 12 attention heads, 768 hidden size and a total paramterers of 110 million. 
To set up our model, we went through the following steps:

- **Step 1:  Data collection** 
Given the provided labelled training data, no data collection or cleaning was required. However, we used ChatGTP to generate additional datapoints which proved quite valuable in increasing the performance of the model in the end.
- **Step 2: Data preprocessing**
To feed our data to the CamemBert model, we needed to apply the Label Encoder to the column "difficulty". Moreover, we split the data into training and evaluation sets (80/20).
- **Step 3: Tokenization**
We then tokenized both our training and evaluation set using the CamemBert tokenizer. The CamemBert tokenizer convers raw text into a numerical format that can be processed by the model.
- **Step 4: Load model**
The CamemBert model is loaed with its pretrained weights. In our case, we load the CamembertFor SequencesClassification which adds an additional linear classification layer to the pretrained model.
- **Step 5: Define Training Parameters**
The Camembert model allows to specifc various training arguments such as batch size, learning rate, training epochs, or weight decay. The combination of these parameters can signifcantly impact both the computational resources required to run the model as well as the performance (see step 7). The training parameters automatically included the AdamW optimizer to optimize the weights of the model and a learning rate scheduler which adjusts the learnin rate as the model is fine-tuned. The default loss function is cross-entropy loss.
- **Step 6: Fine-tuning the model**
Now it is time to fine-tuned the model on our training data using the defined training parameters. Fine-tuning allows to adapt a pre-trained model to perform a specifc taks, in our case to predict the difficulty of French sentences. Fine-tuning hence allows the model to spcialize in the required task. During the fine-tuning process, the model loops through several key steps in each epoch:
  - forward pass: the inputs (tokzenized text and attention masks) are passed through the model to get predictions.
  - loss calculation: the loss between the predicted and true labels is calculated using cross-entropy.
  - backward pass: the loss is backpropagated to calculate gradients.
  - Parameter update: gradients are used to then update model weigths via the optimizer AdamW.
- **Step 7: Evaluation**
After fine-tuning the model, it was time to evaluate its performance based on the evaluation dataset. Here we primarily use accuracy as the evalution metrics.
- **Step 8: Optimization**
To increase the performance (i.e. accuracy) of the model we set up a hyperoptimization process using optina. This helps us find a good combination of paramerters (epochs, batch size, and learning rate) which we then furhter adjust manually until we are satisfied with the performance level of the model.
- **Step 9: Prediction**
Finally, we can use the model to make prediction on the unlabelled test data. For this, we first re-train the model with the optimized parameters on the extended dataset (the one we generated with CHatGTP) and then use this model to make the final predictions. 




# Application of the Model

For the application of our model, we decided to create a Streamlit app that can predict the French difficulty level of books. This tool may be helpful for language teachers who are looking for books that match the language level of their students as well as for people learning French and want to read a book that matches their skills. There is nothing more frustrating than reading a book that is way too difficult. Our app BOOKLY is simple and user-friendly. Upon loading the app, users can find the input field in the sidebar on the left. To predict the diffictulty level of a book, all they have to do is enter the title of the book and upload an excerpt of the book in docx format. Our app will then input this text into the ML model and output the difficulty level of the text. 
Each inputed text will be automatically saved to the app's library. When clicking on the library button in the sidebar on the left, users can brows through all previously uploaded textb excerpts. The search can be filtered by title or language level. In this way, users can either check for the difficulty level of specific books without having to upload them themselves or search for books that match their language level. The value of the app hence grows over time as more and more people upload excerpts of different books to the app, enlarging the library.

Explore our app: https://machinelearning-bookly.streamlit.app/

# Final Words

Watch our video here: 
