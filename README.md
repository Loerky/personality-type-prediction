# Personality Type Prediction

This project predicts the personality type of a person from a short questionnaire.
It is my machine learning project for the AI/ML Engineering course at MSIT.

A user answers 19 personality questions (1 = Disagree, 5 = Agree) and gives age, gender and writing hand.
The model predicts one of four types:

| Type | Short description |
|---|---|
| Moderate | Balanced, no extreme traits |
| Resilient | Emotionally stable, calm under pressure |
| Overcontroller | Anxious and introverted |
| Undercontroller | Impulsive, less focused on rules |

## Data

The data comes from the Big Five (OCEAN) personality test on Kaggle.
The course prepared a cleaned version with the 19 most useful questions, some demographic data and the target.

- `data.csv` – the dataset (19,719 rows, 23 columns)
- `codebook.txt` – explains all columns and how the target was made

Note: the codebook also describes a type "Reserved", but this type does not appear in the data.

## Project structure

```
├── codebook.txt
├── data.csv
├── 01_eda_and_pipeline.ipynb
├── 02_evaluation_and_tuning.ipynb
├── 03_tuning_and_tracking.ipynb
├── app.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Setup

I used Python 3.13 on Windows.

1. Clone the repository and open the folder.
2. Create and activate a virtual environment:
   ```
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
   (On Mac/Linux: `source .venv/bin/activate`)
3. Install the libraries:
   ```
   pip install -r requirements.txt
   ```

## How to run

Run all commands from the project folder.

1. Run the notebooks in order:
   - `01_eda_and_pipeline.ipynb` – data check, cleaning, preprocessing pipeline, first models
   - `02_evaluation_and_tuning.ipynb` – classification report, cross-validation, testing max_depth
   - `03_tuning_and_tracking.ipynb` – compare models, tuning with Hyperopt, MLflow tracking, final model

2. Start the MLflow UI. Use the absolute path to `mlflow.db` in your project folder
   (the notebook 03 prints it as "Tracking URI"):
   ```
   mlflow ui --backend-store-uri "sqlite:///C:/path/to/your/project/mlflow.db"
   ```
   (On Mac/Linux the path starts with `/`, so you get four slashes: `sqlite:////path/to/your/project/mlflow.db`)

   Open http://127.0.0.1:5000, choose "Model training" and the experiment `personality-type-prediction`.

3. Copy the Run ID of the run `xgb-final` and paste it in `app.py` (`RUN_ID = "..."`).
   Every new run gets a new ID, so this step is needed after you run notebook 03.

4. Start the Streamlit app:
   ```
   streamlit run app.py
   ```

## Results

Model comparison (F1-macro, 5-fold cross-validation on the train data, default settings):

| Model | F1-macro |
|---|---|
| Random Forest | 0.743 |
| Logistic Regression | 0.768 |
| Logistic Regression (balanced) | 0.784 |
| Random Forest (balanced) | 0.784 |
| XGBoost | 0.796 |

I tuned XGBoost with Hyperopt (20 tries), but the best result (0.794) was not better than the default settings.
So the final model is XGBoost with default settings (300 trees, max_depth 4, learning rate 0.1).

Final model on the test data:
- F1-macro: 0.803
- Accuracy: 0.866
- Undercontroller is the hardest type (F1 0.50). For this type the data has less information.

F1-macro is the main metric, because the classes are not balanced.

## Data cleaning

- Ages over 100 are not realistic (birth years, wrong numbers). I set them to missing.
- A 0 in the questions means no answer, so I set it to missing.
- 3 duplicate rows are removed.
- Missing values are filled inside the pipeline (median for numbers, most frequent value for gender and hand).
