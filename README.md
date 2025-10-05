# Sentiment-Analysis-

## 🎬 Project Demonstration

<p align="center">
  <img src="your_demo_video_or_gif_link_here" alt="Sentiment Analysis Web App Demo" width="700">
</p>

*A short demo showing how a user enters a comment and receives a sentiment prediction.*
## 📖 Project Overview

This project performs **Sentiment Analysis**, classifying user-entered comments as either **Positive** or **Negative**.  

It demonstrates an **end-to-end machine learning workflow**, from raw text to a deployed web application, including:
- Data collection and exploration
- Text cleaning and preprocessing
- Feature engineering and class balancing
- Model training and evaluation
- Prediction pipeline development
- Deployment via a Flask web application
## ⚙️ Project Setup

To maintain a clean working environment, a **virtual environment** was created. All necessary packages were installed in this environment to ensure reproducibility.

**Packages used for the project:**
- NumPy, Pandas, Matplotlib – for data handling, exploration, and visualization  
- NLTK – for NLP tasks such as stopword removal and stemming  
- Scikit-learn – for machine learning algorithms and evaluation  
- Imbalanced-learn (SMOTE) – for handling class imbalance  
- Flask – to build a simple web interface  
- Pickle – to save and load trained models
## 🗄️ Data Collection

- Dataset sourced from **Kaggle**  
- **7,920 records**, 3 columns:
  - `ID` – Unique identifier for each comment  
  - `Label` – Target variable indicating sentiment (`Positive` or `Negative`)  
  - `Comment` – Raw text input  

- Initial exploration confirmed there were **no missing or duplicate values**.
## 🧹 Data Cleaning and Preprocessing

The comments contained messy text. The following NLP preprocessing steps were applied:

1. **Lowercase conversion** – all words converted to lowercase.  
2. **Remove links/URLs** – identified words starting with `https` using regex and removed them.  
3. **Remove punctuation** – cleaned all punctuation using Python’s `string` module.  
4. **Remove numbers** – numeric characters were removed.  
5. **Remove stopwords** – words with no semantic value (e.g., “is”, “am”, “are”) were removed using NLTK’s English stopword list.  
6. **Stemming** – reduced words to their root form using `PorterStemmer` (e.g., “beautiful” → “beauti”).

This ensured the dataset was clean, uniform, and ready for model training.
## 🧠 Feature Engineering

- **Vocabulary Creation:** All unique words were collected using `collections.Counter`. Original vocabulary: 15,960 words.  
- **Dimensionality Reduction:** Words appearing fewer than 10 times were removed to avoid overfitting. Final vocabulary: 1,169 words.  
- Vocabulary was saved for future vectorization.  
- Comments were transformed into **numerical vectors** using the saved vocabulary for model input.
  
## 6. Data Splitting and Class Balancing

The dataset was split into training and testing sets (80% training, 20% testing):

- **Training:** 6,336 records
- **Testing:** 1,584 records

### Class Imbalance

- **Negative:** 4,732  
- **Positive:** 1,604  

SMOTE was applied to balance the classes in the training set, resulting in:

- **Positive:** 4,732  
- **Negative:** 4,732  

This step ensured the model would not be biased toward the majority class.


## 🤖 Model Building and Evaluation

**Algorithms Tested:**  
- Logistic Regression  
- Naive Bayes (MultinomialNB)  
- Decision Tree  
- Random Forest  
- Support Vector Machine (SVM)  

**Evaluation Metrics:**  
- Accuracy – overall correctness of the model  
- Precision – correctness of positive predictions  
- Recall – ability to identify all positive instances  
- F1-Score – harmonic mean of precision and recall  

Training and testing accuracies were compared to check for **overfitting**.

## 🏆 Final Model Selection

The **Naive Bayes (MultinomialNB)** model was selected because:  
- High accuracy on both training and testing datasets  
- Minimal difference between training and testing accuracy  
- Well-suited for high-dimensional text classification

The trained model was saved using **Pickle**.
## 🧩 Prediction Pipeline

**Steps for predicting user input:**
1. **Preprocessing** – lowercase conversion, remove links, punctuation, numbers, stopwords, and apply stemming.  
2. **Vectorization** – transform cleaned text into numerical vectors using the saved vocabulary.  
3. **Prediction** – vectorized text is passed to the Naive Bayes model to output sentiment (Positive or Negative).
## 🌐 Web Application (Flask Deployment)

- Built using **Flask** for a simple web interface.  

**Workflow:**  
1. User enters a comment in the input box  
2. Comment passes through the **prediction pipeline**  
3. Sentiment prediction (Positive or Negative) is displayed on the page  


## ☁️ Future Work

- Deploy the application on **Microsoft Azure** for public access.  
- Increase dataset size to improve model performance.  
- Explore deep learning-based NLP models (e.g., LSTM, BERT).  
- Add interactive visualizations like word frequency or sentiment distribution.

