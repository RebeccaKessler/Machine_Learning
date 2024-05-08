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


# Application of the Model

For the application of our model, we decided to create a Streamlit app that can predict the French difficulty level of books. This tool may be helpful for language teachers who are looking for books that match the language level of their students as well as for people learning French and want to read a book that matches their skills. There is nothing more frustrating than reading a book that is way too difficult. Our app BOOKLY is simple and user-friendly. Upon loading the app, users can find the input field in the sidebar on the left. To predict the diffictulty level of a book, all they have to do is enter the title of the book and upload an excerpt of the book in docx format. Our app will then input this text into the ML model and output the difficulty level of the text. 
Each inputed text will be automatically saved to the app's library. When clicking on the library button in the sidebar on the left, users can brows through all previously uploaded textb excerpts. The search can be filtered by title or language level. In this way, users can either check for the difficulty level of specific books without having to upload them themselves or search for books that match their language level. The value of the app hence grows over time as more and more people upload excerpts of different books to the app, enlarging the library.

Explore our app: https://machinelearning-bookly.streamlit.app/

# Final Words

Watch our video here: 
