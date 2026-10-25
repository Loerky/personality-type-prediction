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

Run the notebook `01_eda_and_pipeline.ipynb`. It checks and cleans the data, builds the preprocessing pipeline and trains two first models.

## Results so far

| Model | Accuracy |
|---|---|
| Random Forest | 0.842 |
| Logistic Regression | 0.850 |

## Data cleaning

- Ages over 100 are not realistic (birth years, wrong numbers). I set them to missing.
- A 0 in the questions means no answer, so I set it to missing.
- 3 duplicate rows are removed.
- Missing values are filled inside the pipeline (median for numbers, most frequent value for gender and hand).
