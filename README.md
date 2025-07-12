# 📰 Fake News Detection using NLP and Logistic Regression

This project uses **Natural Language Processing (NLP)** techniques and a **Logistic Regression** model to classify news articles as either real or fake. The model learns to detect fake news based on text data, including the author, title, and content of articles.

---

## 📄 Dataset

- **Source:** [Source-Based News Classification Dataset](https://www.kaggle.com/datasets/ruchi798/source-based-news-classification)
- **Description:** Contains news articles with the following columns:
  - `id`: Unique article ID
  - `title`: Title of the article
  - `author`: Author of the article
  - `text`: Main text of the article (may be incomplete)
  - `label`: 1 for fake news, 0 for real news

---

## ⚙️ Technologies Used

- Python
- NumPy
- Pandas
- NLTK
- Scikit-learn

---

## 🚀 Project Workflow

### 1️⃣ Data Loading & Exploration
- Load the dataset from Kaggle using `kagglehub`.
- Explore the shape, check for missing values, and preview data samples.

### 2️⃣ Data Preprocessing
- Merge author names and titles to form a combined `content` column.
- Replace missing values with empty strings.
- Apply text cleaning, stopword removal, and stemming using `nltk`'s `PorterStemmer`.
- Convert cleaned text data into numerical features using **TF-IDF Vectorization**.

### 3️⃣ Data Splitting
- Split the dataset into training and test sets (80% training, 20% test) using `train_test_split`.

### 4️⃣ Model Training
- Train a **Logistic Regression** classifier on the training set to learn patterns distinguishing real and fake news.

### 5️⃣ Model Evaluation
- Evaluate performance using **accuracy score** on both training and test sets.
- Check for overfitting and generalization capability.

### 6️⃣ Prediction on New Data
- Test the model by predicting on unseen samples and interpret the results as real or fake.

---

## ✅ Results

- **Training Accuracy:** ~89.0%, indicating the model has learned important patterns from the training data without overfitting too much.
- **Test Accuracy:** ~80.2%, suggesting the model generalizes well and performs effectively on unseen news articles.

---

## 💡 What We Learned

- How to preprocess and clean text data using NLP techniques.
- Applying TF-IDF vectorization to convert textual data into numerical form.
- Training and evaluating a Logistic Regression model for binary text classification.
- Building a simple predictive system for real-world fake news detection.

---

## 📥 How to Run

1️⃣ **Clone this repository:**

```bash
git clone https://github.com/RONAKBAGRI/Fake-News-Detection-using-NLP-and-Logistic-Regression.git
```

2️⃣ **Install dependencies:**
```bash
pip pip install numpy pandas nltk scikit-learn kagglehub
```

3️⃣ **Run the notebook:**
```bash
jupyter notebook Fake_News_Prediction.ipynb
```