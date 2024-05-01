# Machine_Learning


## Welcome to final report of the Group Swatch 

In this report, you can find the following sections:
- [Team Approach and Task Division](#team-Approach-and-Task-Division)
- [Comparison of Different Models](#Comparison-of-Different-Models)
- [Final Model](#Final-Model)
- [Application of the Model](#Application-of-the-Model)
- [Final Words](#Final-Words)


# Team Approach and Task Division

We approached the task as a reiterative process, incrementally improving our model to predict the language level of French texts. We started off with simple ML models inlcuding Logistic Regression, KNN, Decision Tree, and Random Forest before then advancing to more sophisticated text embeddings. Within the different models, we used hyper-parameter optimization to find the best solution. 
From the beginning, we thereby already started to think about potential applications of our final model, an attractive user interfence, and an engaging concept for the video presentation. 
Hence, we distirbuted the tasks among the members accordingly:
- Giulia: creating the ML model to predict the language level of French texts
- Rebecca: working on the application and UI of the final model as well as preparing the report



# Comparison of Different Models


| Metric    | Logistic Regression | KNN | Decision Tree | Random Forest | Any other technique |
|-----------|---------------------|-----|---------------|---------------|---------------------|
| Precision |                     |     |               |               |                     |
| Recall    |                     |     |               |               |                     |
| F1-score  |                     |     |               |               |                     |
| Accuracy  |                     |     |               |               |                     |

# Final Model

# Application of the Model

For the application of our model, we decided to create a Streamlit app that can predict the French difficulty level of books. This tool may be helpful for language teachers who are looking for books that match the language level of their students as well as for people learning French and want to read a book that matches their skills. There is nothing more frustrating than reading a book that is way too difficult. Our app BOOKLY is simple and user-friendly. Upon loading the app, users can find the input field in the sidebar on the left. To predict the diffictulty level of a book, all they have to do is enter the title of the book and upload an excerpt of the book in docx format. Our app will then input this text into the ML model and output the difficulty level of the text. 
Each inputed text will be automatically saved to the app's library. When clicking on the library button in the sidebar on the left, users can brows through all previously uploaded textb excerpts. The search can be filtered by title or language level. In this way, users can either check for the difficulty level of specific books without having to upload them themselves or search for books that match their language level. The value of the app hence grows over time as more and more people upload excerpts of different books to the app, enlarging the library.

Explore our app: https://machinelearning-bookly.streamlit.app/

# Final Words
