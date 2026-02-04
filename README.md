LINK TO DATASET FILE : https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset


📌 Fake News Detection – Workflow

STEPS FOLLOWED :
* Fake News Detection
* Importing Libraries
* Importing Dataset
* Inserting a Target Column (`class`)
* Merging True and Fake DataFrames
* Creating a Text Preprocessing Function
* Defining Dependent and Independent Variables
* Splitting Training and Testing Sets
* Converting Text to Vectors
* Logistic Regression
* Decision Tree Classification
* Gradient Boosting Classifier
* Random Forest Classifier
* Model Testing

📰 Fake News Detection

This project focuses on detecting fake news articles using **Natural Language Processing (NLP)** and **Machine Learning** techniques. The workflow follows a structured pipeline from data loading to model evaluation.



📥 Importing Libraries

Essential Python libraries such as NumPy, Pandas, NLTK, and Scikit-Learn are imported to support data manipulation, text preprocessing, feature extraction, and model training.



📊 Importing Dataset

The dataset containing fake and real news articles is loaded using Pandas. The data is prepared for further preprocessing and analysis.



🏷️ Inserting a Target Column (`class`)

A target column named `class` is added to the dataset:

* `0` → Fake News
* `1` → Real News

This column acts as the label for supervised learning.


🔗 Merging True and Fake DataFrames

The fake and real news datasets are combined into a single DataFrame to create a unified dataset for training the model.

---

🧹 Creating a Text Preprocessing Function

A preprocessing function is defined to:

* Convert text to lowercase
* Remove punctuation and special characters
* Remove stopwords
* Apply lemmatization

This improves model accuracy by cleaning the textual data.



🎯 Defining Dependent and Independent Variables

* Independent Variable (X):** News article text
* Dependent Variable (Y):** Class label (Fake or Real)



✂️ Splitting Training and Testing Sets

The dataset is split into training and testing sets using an 80:20 ratio to evaluate model performance on unseen data.



🔢 Converting Text to Vectors

Text data is transformed into numerical form using **TF-IDF Vectorization**, enabling machine learning models to process text effectively.



🤖 Logistic Regression

A Logistic Regression model is trained to classify news articles and serves as a baseline classifier.

 
🌳 Decision Tree Classification

A Decision Tree model is used to learn hierarchical patterns in the data for classification.



 🚀 Gradient Boosting Classifier

Gradient Boosting improves performance by combining multiple weak learners into a strong predictive model.


🌲 Random Forest Classifier

The Random Forest model uses multiple decision trees to enhance accuracy and reduce overfitting.



✅ Model Testing

All trained models are evaluated using accuracy score and classification reports to compare performance and select the best model.

