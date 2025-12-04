# ML Classifier and Hybrid Recommender for Cold-Start Language App Discovery
---
This project builds a hybrid, cold-start recommendation system designed to help users discover effective language-learning apps—even when no behavioral or historical user data is available. Traditional approaches based solely on popularity rankings or manual categories often fail to capture what actually makes an app valuable for learning. This system instead combines statistical quality signals, predictive modeling, and content-based features to surface apps with both proven reliability and strong future potential.

## Background
---
I previously built [Finny](https://github.com/erena0424/finny-demo), an online database that helps users find apps through keyword search, categories, and recommendations. This system relied on manual categories and collaborative filtering, which was flawed because: 

1. **Manucal categorization was arbitrary**: Manually assigned categories were subjective and hard to scale, making consistent classification across a large set of apps impossible.
2. **Cold start failuare**: The recommendation system was useless without user data.

This new project addresses these flaws by building a scalable, data-driven system to categorize and rank apps even before any user clicks a button.


## Project Pipeline (3 Components)
---
The entire workflow is split across three sequential files, demonstrating a clean separation between data acquisition, feature engineering, and model application.

## 0. Getting Started
---
### 0-1. Environment Setup

Create and activate a virtual environment (recommended):
```
conda create -n env python=3.9.6
conda activate env
```

### 0-2. Install Dependencies
Install the required packages listed in your requirements.txt:
```
pip install -r requirements.txt
```

### 0-3. Download NLP Resources

The NLTK library requires downloading internal linguistic data (stopwords, wordnet) once. Run the following command in a Python shell or within the first cell of the notebooks:
```
import nltk
nltk.download(['punkt', 'stopwords', 'wordnet'])
```


## 1. Data Sourcing
----
[00_Data_Acquisition_and_Quality_Filtering.ipynb](00_Data_Acquisition_and_Quality_Filtering.ipynb)
### Objective
Acquire a high-signal dataset of language apps, remove noise, and perform essential low-level ETL (Extract, Transform, Load) tasks.

### Key Deliverables
* **Targeted Scraping:** Used expanded search queries to maximize recall of relevant apps.
* **Initial Feature Engineering:** Applied robust cleaning to convert raw strings (e.g., install counts) into usable numerical data.
* **Filtering:** Applied domain-specific keyword filtering to ensure every app is relevant, removing irrelevant general 'Education' content.

## 2. EDA, Feature Engineering, and App Categorization
----
[01_EDA_App_Categorization_and_Features.ipynb](01_EDA_App_Categorization_and_Features.ipynb)

### Objective
Validate data quality, create the core methodology feature (`category`), and prove the necessity of moving to a supervised model.

### Key Deliverables
* **Exploratory Data Analysis (EDA):** Confirmed the volatility of the raw `score` (justifying the **Bayesian Average** correction in the next stage).
* **Unsupervised Experimentation:** Tested K-Means, Fuzzy C-Means (FCM), and HDBSCAN. The consistent failure of these models due to data sparsity validated the ultimate need for a **Supervised Classification** solution.
* **Supervised Feature Flags:** Created all binary/boolean features (e.g., `has_ai_tutor`, `is_big_developer`) needed for the predictive model.
* **Final Output:** A high-quality feature set ready for supervised model training.


*(language_apps.csv)[language_apps.csv] has files saved from this process.
*[language_apps_MANUAL_CATEGORY_UPDATE.csv](language_apps_MANUAL_CATEGORY_UPDATE.csv) has some manually updated categories for supervised learning.

## 3. Recommender Building
----
[02_Popularity_Prediction_and_Hybrid_Recommender.ipynb](02_Popularity_Prediction_and_Hybrid_Recommender.ipynb)

### Objective
Apply the Supervised Learning-to-Rank (LTR) concept to synthesize two distinct ranking factors into a single score.

### Key Deliverables

#### A. Predictive Analysis (Feature Alignment Score - FAS)
* **Model:** Used a high-performance **XGBoost Classifier** (with cross-validation) to predict **High Popularity** (Quality $\ge 4.0 \text{ AND } \text{Installs} \ge 500k$).
* **FAS Generation:** The **Feature Importances** (weights) derived from XGBoost are used to calculate the **FAS**, which assigns a predictive score to each app.

#### B. Hybrid Ranking Formula
* **W1 (Quality):** **Bayesian Adjusted Score** (BAS) — Corrects statistical volatility in raw scores.
* **W2 (Predictive):** **Normalized FAS** — Provides a predictive boost to apps with high-potential features (e.g., AI, high max IAP price).
* **Final Result:** The recommender ranks apps based on the linear combination: $\text{Rank} = (W_1 \cdot \text{BAS}) + (W_2 \cdot \text{FAS})$.

