# 📱 Megaline Mobile Plan Recommendation
### *Turning Subscriber Behavior into Smarter Plan Choices*

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2.2-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Wrangling-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://megaline-mobile-plan-recommendation-levitoncarvalho.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](https://opensource.org/licenses/MIT)

<br>

**[🚀 Try the Live App](https://megaline-mobile-plan-recommendation-levitoncarvalho.streamlit.app/)** &nbsp;|&nbsp; **[📓 View Full Notebook](https://github.com/levitoncarvalho/megaline-mobile-plan-recommendation/blob/main/notebooks/)**

</div>

---

> ⚠️ **Disclaimer:** Megaline is a **fictional telecom operator** created exclusively for academic and portfolio purposes. This project was developed as part of a Data Science training program and is intended solely to demonstrate technical skills in machine learning, data analysis, and model deployment. No real telecom company, subscriber data, or business relationship is represented here.

---

## 🧩 The Business Problem

> *"The right plan, at the right time, for the right customer — that's the difference between loyalty and churn."*

**Megaline**, a fictional mobile carrier, had a growing problem: a large portion of its subscribers were still using **legacy plans** — outdated packages that no longer matched their usage patterns or needs.

The company wanted a smarter approach: instead of pushing generic offers to everyone, they needed a model capable of **analyzing each subscriber's behavior** and recommending the most suitable modern plan — either **Smart** or **Ultra**.

### Why Does This Matter?

| Scenario | Impact |
|---|---|
| 🔴 No recommendation model | Subscribers stay on legacy plans indefinitely, reducing satisfaction and revenue |
| 🟡 Weak model (Accuracy < 0.75) | Incorrect recommendations damage customer trust |
| 🟢 **Optimized model (Accuracy = 0.7963)** | **Data-driven plan matching that benefits both the customer and the business** |

---

## 📊 Final Results

<div align="center">

### 🏆 The final model exceeded the business accuracy threshold

</div>

| Metric | Target | Result | Status |
|---|---|---|---|
| **Accuracy** (test set) | ≥ 0.75 | **0.7963** | ✅ Exceeded |
| **Algorithm** | — | Random Forest | ✅ Successful |
| **Validation Strategy** | — | Train / Validation / Test Split | ✅ Successful |
| **Deployment** | — | Streamlit Cloud | ✅ Successful |

> **An accuracy of ~0.80 means the model correctly recommends the right plan for 4 out of every 5 subscribers** — reliable enough to power real-time personalized recommendations at scale.

---

## 🧠 End-to-End Data Science Workflow

```
📥 Raw Data  →  🔎 EDA  →  🛠️ Preprocessing  →  🤖 Modeling  →  🎯 Evaluation  →  🚀 Deployment
```

### 1. 🔎 Exploratory Data Analysis (EDA)

The dataset used in this project contains **behavioral data from subscribers who have already switched to a modern plan**, providing a clean, labeled foundation for classification.

- **Dataset:** `user_behavior.csv` — subscriber usage records
- **No missing values** — clean dataset, no imputation required
- **Target variable:** plan label (`0` = Smart, `1` = Ultra)
- **No class imbalance treatment needed** — dataset was reasonably balanced between plan classes

**Feature Overview:**

| Feature | Type | Description |
|---|---|---|
| `calls` | Numerical | Number of calls made per month |
| `minutes` | Numerical | Total call duration in minutes per month |
| `messages` | Numerical | Number of text messages sent per month |
| `mb_used` | Numerical | Mobile data used per month (in MB) |

**Key behavioral patterns discovered:**
- Subscribers with higher `mb_used` and `minutes` tend to benefit more from the **Ultra** plan
- Lower usage across all features correlates strongly with the **Smart** plan
- `mb_used` emerged as the most discriminating feature for plan classification

---

### 2. 🤖 Modeling & Optimization

Three algorithms were trained and evaluated on the validation set, with hyperparameter tuning applied to select the best configuration:

| Algorithm | Validation Accuracy | Notes |
|---|---|---|
| **Random Forest** | **Best** | ✅ Chosen model |
| Decision Tree | Competitive | Slightly lower generalization |
| Logistic Regression | Lower | Limited by linear decision boundary |

**Why Random Forest won:** Its ensemble nature reduces variance compared to a single Decision Tree, and it naturally handles non-linear relationships between usage features — making it better at distinguishing nuanced subscriber profiles.

- **Optimization:** Systematic hyperparameter tuning across `n_estimators`, `max_depth`, `min_samples_split`, and `min_samples_leaf`
- **Evaluation split:** Training → Validation → Test (three-way holdout)
- **Primary metric:** Accuracy

---

### 3. 🎯 Final Test Set Evaluation

The final model was evaluated on data **never seen during training or validation**:

```
Accuracy  = 0.7963  ✅  (target: ≥ 0.75)
```

The model correctly classifies the right plan for nearly **80% of subscribers** on unseen data — demonstrating solid generalization and making it suitable for real-world recommendation workflows.

---

### 4. 🚀 Deployment

The trained model was serialized with `joblib` and deployed as an interactive **Streamlit web app**, allowing users to simulate a subscriber's usage profile via sliders and receive an instant plan recommendation — **Smart** or **Ultra** — with a clear explanation of why that plan fits.

**[➡️ Try the live app here](https://megaline-mobile-plan-recommendation-levitoncarvalho.streamlit.app/)**

---

## 🗂️ Project Structure

```text
megaline-mobile-plan-recommendation/
│
├── 📂 data/
│   └── 📊 user_behavior.csv                        # Raw subscriber behavior dataset
│
├── 📓 notebooks/
│   └── 📓 exploration_v1.ipynb                     # Full analysis + modeling
│
├── 🤖 models/
│   └── ⚙️ best_model.pkl                           # Serialized trained model
│
├── 🐍 src/
│   ├── 🐍 __init__.py                              # Package initialization
│   ├── 🐍 data.py                                  # Data loading & splitting
│   ├── 🐍 train.py                                 # Model training & tuning
│   └── 🐍 predict.py                               # Inference helper functions
│
├── 🌐 app.py                                       # Streamlit interactive app
├── 🔁 train_and_test.py                            # Full training & evaluation pipeline
├── 📦 requirements.txt                             # Python dependencies
├── 🚫 .gitignore                                   # Ignored files and folders
├── ⚖️ LICENSE                                      # MIT License
└── 📄 README.md                                    # Project documentation
```

---

## 🚀 Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/levitoncarvalho/megaline-mobile-plan-recommendation.git
cd megaline-mobile-plan-recommendation

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train and evaluate the model
python train_and_test.py

# 5. Launch the app
streamlit run app.py
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.9+ |
| **Data Manipulation** | Pandas, NumPy |
| **Machine Learning** | Scikit-Learn |
| **Visualization** | Matplotlib, Seaborn |
| **Serialization** | Joblib |
| **Deployment** | Streamlit, Streamlit Community Cloud |

---

## 💡 Key Technical Takeaways

- **Accuracy is the right metric here** — unlike churn prediction, the Megaline dataset is balanced between plan classes, making accuracy a reliable and interpretable performance measure.
- **Random Forest outperformed both Decision Tree and Logistic Regression** — its ensemble approach better captures the non-linear relationships between usage behavior and plan suitability.
- **A three-way train/validation/test split** ensures honest evaluation: the validation set was used for tuning, and the test set was touched only once — giving a trustworthy final accuracy estimate.
- **The modular `src/` structure** (`data.py`, `train.py`, `predict.py`) promotes clean separation of concerns and makes the codebase easy to maintain, test, and extend.
- **Streamlit deployment** bridges the gap between a Jupyter notebook and a usable product — turning a trained model into a real-time interactive tool accessible to non-technical stakeholders.

---

## 👨‍💻 Author

<div align="center">

**Leviton Lima Carvalho**
*Data Scientist | Machine Learning | Python*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-levitoncarvalho-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/levitoncarvalho/)
[![GitHub](https://img.shields.io/badge/GitHub-levitoncarvalho-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/levitoncarvalho)
[![Email](https://img.shields.io/badge/Email-levitoncarvalho@icloud.com-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:levitoncarvalho@icloud.com)

</div>

---

## 📄 License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for more details.
