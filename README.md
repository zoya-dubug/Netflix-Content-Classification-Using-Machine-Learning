# Netflix Content Classification Using Machine Learning

## Overview

This project analyzes the Netflix Titles dataset and builds machine learning models to classify content as either a **Movie** or a **TV Show**.

The project includes data cleaning, exploratory data analysis (EDA), feature engineering, model training, and performance evaluation using multiple classification algorithms.

---

## Dataset

Dataset: Netflix Movies and TV Shows Dataset

The dataset contains information such as:

* Title
* Director
* Cast
* Country
* Release Year
* Rating
* Genre (listed_in)
* Content Type (Movie / TV Show)

---

## Project Workflow

### 1. Data Cleaning

* Handled missing values in:

  * Director
  * Country
  * Cast
* Removed rows with missing values in required machine learning features.

### 2. Exploratory Data Analysis (EDA)

Performed analysis and visualization of:

* Movies vs TV Shows distribution
* Rating distribution
* Release year distribution
* Top Netflix genres

Libraries used:

* Pandas
* Matplotlib
* Seaborn

### 3. Feature Engineering

Selected and transformed features including:

* Rating
* Release Year
* Director

Applied Label Encoding to convert categorical variables into numerical form.

### 4. Model Training

Trained and evaluated:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier

### 5. Evaluation

Performance was evaluated using:

* Accuracy Score
* Confusion Matrix

---

## Results

### Experiment 1

Features:

* Rating
* Release Year

| Model               | Accuracy |
| ------------------- | -------- |
| Decision Tree       | 70.98%   |
| Random Forest       | 70.53%   |
| Logistic Regression | 68.82%   |

### Experiment 2

Features:

* Rating
* Release Year
* Director

| Model               | Accuracy |
| ------------------- | -------- |
| Decision Tree       | 93.02%   |
| Random Forest       | 93.98%   |
| Logistic Regression | 86.66%   |

### Observation

Adding director information significantly improved classification performance, indicating a strong relationship between directors and content type.

Genre information was also tested and produced near-perfect accuracy because it was highly correlated with the target variable (Movie vs TV Show).

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

## Project Structure

Netflix_ML_Project/

├── netflix_titles.csv

├── eda.py

├── train_model.py

├── README.md

└── screenshots/

---

## Future Improvements

* Use additional metadata features
* Perform advanced feature engineering
* Explore recommendation systems
* Handle high-cardinality features more effectively
* Deploy the model as a web application

---

## Author

Second-year B.Tech student project focused on learning practical machine learning, data analysis, and model evaluation using real-world datasets.
