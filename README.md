# 🚨 Disaster Tweet Classification Using NLP and Machine Learning

## 📌 Project Overview

Social media platforms such as Twitter are widely used to share information during emergencies and disasters. However, not every tweet containing disaster-related words actually describes a real disaster.

For example, words such as *fire*, *bomb*, *flood*, or *earthquake* may also appear in jokes, metaphors, entertainment, or everyday conversations.

This project develops a Natural Language Processing (NLP) and Machine Learning system that automatically classifies tweets into:

- **0 — Non-Disaster**
- **1 — Disaster**

Multiple machine learning models were trained, evaluated, validated, and tuned. The final selected model was deployed through an interactive **Streamlit dashboard**.

---

# 🎯 Project Objectives

The main objectives of this project are to:

- Explore and understand the disaster tweet dataset.
- Clean and preprocess noisy social-media text.
- Extract useful textual features using TF-IDF.
- Train multiple machine learning classification models.
- Compare models using suitable evaluation metrics.
- Perform cross-validation and hyperparameter tuning.
- Analyze false-positive and false-negative predictions.
- Select the best-performing model.
- Save the trained model for reuse.
- Build an interactive Streamlit web interface.
- Organize the project using reusable Python modules.

---

# 📊 Dataset Overview

The dataset contains **7,613 tweets**.

| Information | Value |
|---|---:|
| Total Tweets | 7,613 |
| Total Columns | 5 |
| Non-Disaster Tweets | 4,342 |
| Disaster Tweets | 3,271 |
| Unique Tweet Texts | 7,503 |
| Unique Keywords | 221 |
| Unique Locations | 3,293 |

### Dataset Columns

The original dataset contains:

- `id`
- `keyword`
- `location`
- `text`
- `target`

The `target` column represents the class label.

- `0` = Non-Disaster
- `1` = Disaster

---

# 📁 Project Structure

```text
Disaster_Tweet_Classification/
│
├── data/
│   └── twitter_disaster.xlsx
│
├── notebooks/
│   └── disaster_tweet_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── evaluation.py
│   ├── validation.py
│   ├── model_persistence.py
│   └── prediction.py
│
├── models/
│   └── disaster_tweet_svm_pipeline.pkl
│
├── reports/
│   ├── figures/
│   ├── tuned_model_validation.csv
│   ├── final_tuned_model_results.csv
│   ├── false_positives.csv
│   └── false_negatives.csv
│
├── dashboard.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🔍 Part 1 — Data Exploration and Preparation

## Task 1.1 — Data Exploration

Exploratory Data Analysis (EDA) was performed to understand the structure and characteristics of the dataset.

The analysis included:

- Dataset shape
- Column information
- Data types
- Missing values
- Duplicate records
- Target-class distribution
- Tweet-length distribution
- Keyword analysis
- Word-frequency analysis

### Target Distribution

The dataset contains:

- **4,342 Non-Disaster tweets — 57.03%**
- **3,271 Disaster tweets — 42.97%**

This indicates a moderate class imbalance but not an extreme one.

---

## Task 1.2 — Data Preparation

Tweet text contains noise that may negatively affect machine learning models.

The preprocessing pipeline therefore performs:

- Missing-value handling
- Duplicate removal
- Lowercase conversion
- URL removal
- HTML removal
- User mention removal
- Hashtag-symbol removal while retaining hashtag words
- Special-character removal
- Extra whitespace removal
- Stopword removal
- Lemmatization

Reusable preprocessing functions were implemented inside:

```text
src/preprocessing.py
```

This allows the same preprocessing logic to be reused during both model development and deployment.

---

# 🧠 Part 2 — Feature Engineering and Model Selection

## Task 2.1 — Feature Engineering

Several numerical features were initially explored, including:

- Tweet length
- Word count
- Hashtag count
- Mention count
- URL count
- Exclamation-mark count
- Question-mark count
- Sentiment polarity

For text representation, **TF-IDF (Term Frequency–Inverse Document Frequency)** was used.

The tuned final text representation used:

```text
ngram_range = (1, 2)
```

This allows the model to learn both:

- Unigrams — individual words
- Bigrams — two-word combinations

TF-IDF was fitted only on training data to prevent data leakage.

---

## Task 2.2 — Model Selection and Training

The following machine learning algorithms were investigated:

1. Logistic Regression
2. Linear Support Vector Machine
3. Multinomial Naive Bayes
4. Random Forest

The models were compared using several classification metrics.

---

# 📈 Part 3 — Model Evaluation and Validation

## Task 3.1 — Initial Model Evaluation

Initial test-set performance was:

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 80.37% | 80.55% | 71.56% | 75.79% |
| Linear SVM | 78.73% | 76.03% | 73.70% | 74.84% |
| Random Forest | 79.19% | 85.03% | 62.54% | 72.07% |
| Multinomial Naive Bayes | 82.01% | 87.85% | 67.43% | 76.30% |

Additional evaluation included:

- Classification reports
- Confusion matrices
- ROC-AUC
- Precision-Recall analysis
- Average Precision

---

# 🔁 Task 3.2 — Model Validation

To obtain a more reliable estimate of model performance, **5-fold Stratified Cross-Validation** was performed.

TF-IDF was placed inside Scikit-learn pipelines during cross-validation so that the vectorizer was fitted independently within each training fold.

This helps prevent vocabulary leakage.

### Cross-Validation Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 79.93% | 82.82% | 67.24% | 74.22% | 85.86% |
| Linear SVM | 77.37% | 75.14% | 70.72% | 72.86% | 83.73% |
| Multinomial Naive Bayes | 79.68% | 85.32% | 63.68% | 72.92% | 85.52% |

---

# ⚙️ Hyperparameter Tuning

GridSearchCV was used to identify improved model configurations.

### Logistic Regression

Best parameters:

```text
C = 0.5
class_weight = balanced
ngram_range = (1, 2)
```

Best validation F1:

```text
75.29%
```

### Multinomial Naive Bayes

Best parameters:

```text
alpha = 1.0
fit_prior = False
ngram_range = (1, 1)
```

Best validation F1:

```text
75.11%
```

### Linear SVM

Best parameters:

```text
C = 0.1
class_weight = balanced
ngram_range = (1, 2)
```

Best validation F1:

```text
75.29%
```

---

# 📊 Tuned Model Validation Comparison

| Model | Training F1 | Validation F1 | F1 Std | Train–Validation Gap |
|---|---:|---:|---:|---:|
| Logistic Regression | 85.22% | 75.29% | 1.64% | 9.93 pp |
| Multinomial Naive Bayes | 85.19% | 75.11% | 1.76% | 10.08 pp |
| Linear SVM | 87.03% | 75.29% | 1.63% | 11.74 pp |

Hyperparameter tuning improved the cross-validation F1 scores of all three models.

---

# 🏆 Final Holdout Evaluation

After model selection and tuning, the tuned models were evaluated on the held-out test dataset.

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Tuned Logistic Regression | 80.89% | 78.05% | 77.22% | 77.63% | 86.98% |
| Tuned Multinomial Naive Bayes | 79.51% | 77.23% | 74.16% | 75.66% | 86.41% |
| **Tuned Linear SVM** | **81.09%** | **78.33%** | **77.37%** | **77.85%** | **87.09%** |

## 🥇 Final Selected Model

The **Tuned Linear SVM** was selected as the final model.

It achieved:

- **Accuracy:** 81.09%
- **Precision:** 78.33%
- **Recall:** 77.37%
- **F1 Score:** 77.85%
- **ROC-AUC:** 87.09%

---

# 🧩 Confusion Matrix

The final Linear SVM confusion matrix was:

```text
[[729, 140],
 [148, 506]]
```

Therefore:

| Prediction Type | Count |
|---|---:|
| True Negatives | 729 |
| False Positives | 140 |
| False Negatives | 148 |
| True Positives | 506 |

The model correctly classified **1,235 out of 1,523** holdout tweets.

---

# 🔬 Error Analysis

Qualitative analysis was performed on false-positive and false-negative predictions.

### False Positives

False positives commonly occurred when disaster-related words such as:

- bomb
- earthquake
- wreck
- spill
- detonation

were used in:

- Figurative language
- Humour
- Entertainment
- Technical contexts

### False Negatives

False negatives commonly occurred when real disaster-related tweets contained:

- Indirect wording
- Abbreviations
- Noisy text
- Limited context
- Uncommon expressions

### Key Insight

TF-IDF with Linear SVM performs well at learning statistical word and phrase patterns, but it has limited ability to understand deeper semantic context, metaphorical language, ambiguity, and unusual phrasing.

---

# 💾 Part 4 — Model Deployment

## Task 4.1 — Model Persistence and Prediction

The final trained pipeline was serialized using `joblib`.

Saved model:

```text
models/disaster_tweet_svm_pipeline.pkl
```

The project contains reusable modules for:

```text
src/model_persistence.py
src/prediction.py
```

`model_persistence.py` handles saving and loading models.

`prediction.py` handles preprocessing and inference for new tweets.

---

# 🖥️ Task 4.2 — Streamlit User Interface

An interactive Streamlit dashboard was developed to provide a user-friendly interface for the trained model.

The dashboard includes:

- Real-time tweet classification
- Sample tweet selection
- Original and processed tweet display
- SVM decision-score analysis
- Model-performance metrics
- Interactive model-comparison charts
- Cross-validation results
- Confusion-matrix information
- False-positive analysis
- False-negative analysis
- Project architecture and methodology

The dashboard is implemented in:

```text
dashboard.py
```

---

# 🧱 Code Modularization

A major objective of the project was to avoid keeping all machine learning logic inside the Jupyter notebook.

Reusable functionality was therefore separated into Python modules:

| Module | Responsibility |
|---|---|
| `data_loader.py` | Dataset loading |
| `preprocessing.py` | Text preprocessing |
| `feature_engineering.py` | Feature generation |
| `model_training.py` | Model creation and training |
| `evaluation.py` | Evaluation metrics |
| `validation.py` | Cross-validation and tuning |
| `model_persistence.py` | Model saving and loading |
| `prediction.py` | Prediction of new tweets |

The notebook is primarily used for experimentation, analysis, visualization, and orchestration.

The Streamlit application reuses the same preprocessing and prediction modules.

---

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy
- NLTK
- TextBlob
- Joblib
- Streamlit
- Jupyter Notebook
- VS Code

---

# 🚀 Running the Project

## 1. Activate the Virtual Environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

## 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

## 3. Run the Streamlit Dashboard

```powershell
streamlit run dashboard.py
```

Streamlit will provide a local address, normally:

```text
http://localhost:8501
```

Open it in a browser to use the application.

---

# ⚠️ Limitations

Although the final model performs well, several limitations remain.

- TF-IDF does not fully understand semantic context.
- Metaphorical language may cause false positives.
- Indirect disaster descriptions may cause false negatives.
- Social-media abbreviations and noisy language remain challenging.
- Linear SVM decision-function values should not be interpreted directly as probability percentages.

More advanced contextual language models could be explored in future work.

---

# 🔮 Future Improvements

Possible future improvements include:

- Transformer-based NLP models
- Contextual word embeddings
- Improved handling of abbreviations and slang
- Additional linguistic features
- Probability calibration
- Larger and more diverse training datasets
- Cloud deployment of the Streamlit application

---

# ✅ Conclusion

This project demonstrates an end-to-end Natural Language Processing and Machine Learning workflow for disaster tweet classification.

The project covers data exploration, preprocessing, feature engineering, model training, evaluation, cross-validation, hyperparameter tuning, error analysis, model serialization, modular Python development, and interactive deployment.

Among the evaluated models, the **Tuned Linear SVM** achieved the strongest final holdout performance with an **F1 score of 77.85%** and **ROC-AUC of 87.09%**.

The final system combines a reusable machine learning pipeline with an interactive Streamlit dashboard, making the project suitable for both technical evaluation and practical demonstration.