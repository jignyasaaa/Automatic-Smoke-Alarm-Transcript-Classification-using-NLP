%%writefile README.md
# 🚨 911 Emergency Call Category Classifier

## 🧠 Project Overview
The **911 Emergency Call Category Classifier** is a machine learning-based web application designed to automatically classify emergency call descriptions into key categories such as **Fire**, **EMS**, and **Traffic**.  
By leveraging **Natural Language Processing (NLP)** and **supervised learning algorithms**, this project aims to assist emergency departments in **prioritizing incidents and allocating resources efficiently**, ultimately saving time and lives.

The project focuses on:
- Building robust ML models for text classification using **TF-IDF features**.
- Evaluating performance across multiple classifiers (Naive Bayes, Logistic Regression, Random Forest).
- Deploying an **interactive Streamlit web application** for real-time emergency call analysis and visualization.

---

## 🗂️ Dataset Overview

### **1️⃣ Dataset – 911 Calls Dataset (Montgomery County, PA)**
The dataset contains more than **100,000 emergency call records**, each labeled with the emergency type and call description.

| Feature | Description |
|----------|--------------|
| **title** | Emergency reason (e.g., FIRE: VEHICLE FIRE, EMS: CARDIAC ARREST) |
| **timeStamp** | Time and date of the call |
| **twp** | Township where the incident occurred |
| **lat / lng** | Geographic coordinates of the call |
| **zip** | Postal code of the call origin |

**Target Classes:**
- **EMS (Emergency Medical Services)**
- **Fire**
- **Traffic**

📁 *Dataset Source:* [Kaggle – 911 Calls Dataset](https://www.kaggle.com/mchirico/montcoalert)

---

### 🧹 Preprocessing Steps
- Removed duplicates and missing values  
- Extracted main emergency category from the “title” column  
- Text cleaning (punctuation, digits, stopwords removal, and lowercasing)  
- Lemmatization for root-word normalization  
- TF-IDF vectorization for feature extraction  

---

## ⚙️ Model Training and Evaluation

### NLP Preprocessing Pipeline

| Step | Description | Libraries Used |
|------|--------------|----------------|
| 1️⃣ Text Cleaning | Removed punctuation, digits, and stopwords | `re`, `nltk` |
| 2️⃣ Lemmatization | Reduced words to their base form | `WordNetLemmatizer` |
| 3️⃣ Feature Extraction | TF-IDF transformation for numerical representation | `sklearn.feature_extraction.text.TfidfVectorizer` |

---

### 🧠 Models Used

| Model | Description | Training Accuracy |
|--------|--------------|------------------|
| **Naive Bayes** | Probabilistic model ideal for word-frequency-based data | 97.3% |
| **Logistic Regression** | Linear classifier suitable for text-based prediction | 98.1% |
| **Random Forest** | Ensemble model capturing non-linear decision boundaries | **99.2%** |

✅ **Observation:** Random Forest outperformed other models with the highest precision and recall, indicating robust performance with minimal overfitting.

---

### 📊 Training Results

| Model | Accuracy | Precision | Recall | F1-Score |
|--------|-----------|-----------|---------|-----------|
| Naive Bayes | 97.3 % | 0.972 | 0.973 | 0.972 |
| Logistic Regression | 98.1 % | 0.981 | 0.981 | 0.981 |
| **Random Forest** | **99.2 %** | **0.992** | **0.992** | **0.992** |

<img width="450" height="280" alt="Confusion Matrix" src="https://github.com/user-attachments/assets/your_confusion_matrix_image_link" />

---

## 📈 Model Accuracy Comparison

The bar chart below compares the accuracy of different classification models.
<img width="900" height="400" alt="image" src="https://github.com/user-attachments/assets/9da7692f-fc05-4d28-bc06-2c576c9f296b" />


## 🧪 Experiments and Analysis

To assess the effectiveness of each model:
- **TF-IDF** was used to represent the textual descriptions numerically.
- All models were evaluated on **accuracy, precision, recall, and F1-score**.
- **Random Forest** provided the best trade-off between accuracy and generalization.
- **Visualization tools** such as confusion matrices and category frequency charts were used to interpret model performance.

### Example Visuals:
- Category distribution chart  
- Model comparison plot  
- Confusion matrix heatmap  

---

## 🌐 Streamlit Web Application

An interactive **Streamlit app** was created to classify 911 call texts in real time.  
The web interface allows users to input emergency call descriptions and instantly see predictions from trained models.

🟢 **Live App:**  
👉 [https://dave-lated-inspectingly.ngrok-free.dev](https://dave-lated-inspectingly.ngrok-free.dev)  
*(If hosted locally: [http://localhost:8501](http://localhost:8501))*

---

### 🧭 How the App Works

#### 📝 **Input:**
Users can type or paste **emergency call descriptions** (e.g., “FIRE: HOUSE SMOKE DETECTED”).

#### ⚙️ **Processing:**
1. Text input is preprocessed using the trained TF-IDF vectorizer.  
2. The selected classifier (Random Forest, Logistic Regression, or Naive Bayes) predicts the category.  
3. Results are visualized as a table and bar chart showing prediction confidence.

#### 📋 **Output:**
| Call Description | Predicted Category |
|------------------|--------------------|
| FIRE: HOUSE SMOKE DETECTED | Fire |
| EMS: CARDIAC ARREST | EMS |
| TRAFFIC: VEHICLE ACCIDENT | Traffic |

---

### 🖥️ How to Run the Streamlit App Locally

#### **Step 1️⃣ – Clone the Repository**
bash
git clone https://github.com/JignyasaLunkad/911-Call-Classifier.git
cd 911-Call-Classifier

### 🧰 Install Dependencies
Before running the project, make sure you have Python 3.8+ installed.  
Install all required libraries by running:
bash
pip install -r requirements.txt

---

#### 🖥️ **Launch Streamlit App**
markdown
## 🖥️ Launch Streamlit App
Run the following command in your terminal to start the Streamlit web app:
bash
streamlit run app.py

## 🚀 Future Scope
- Extend the classifier to handle **multi-label emergencies** (e.g., “FIRE: ACCIDENT INVOLVING INJURY”).
- Integrate **real-time emergency APIs** for live classification and monitoring.
- Deploy as a **mobile or web-based alert system** for emergency centers.
- Explore **Deep Learning models (BERT, RoBERTa)** for more advanced text understanding.

## 🏁 Conclusion
This project successfully demonstrates how **Machine Learning** and **NLP** can automate the classification of emergency call transcripts with high accuracy (up to **99.2%**).  
It highlights that:
- Random Forest delivers strong performance with minimal overfitting.
- TF-IDF remains a simple yet effective feature representation for short text.
- Streamlit provides an interactive and efficient deployment platform.

## 📚 References
1. Kaggle – 911 Calls Dataset  
2. Scikit-learn Documentation  
3. Streamlit Documentation  
4. Manning et al. (2008), *Introduction to Information Retrieval*  
5. Jurafsky & Martin (2020), *Speech and Language Processing*  
6. Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32.

