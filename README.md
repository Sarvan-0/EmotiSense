# 🧠 EmotiSense

**AI-Powered Multi-Class Emotion Classification**

EmotiSense is an NLP machine learning project that predicts the emotion expressed in a piece of text.

## Features

- Classifies text into six emotions.
- Uses TF-IDF for text feature extraction.
- Uses a calibrated Linear SVM classifier.
- Displays prediction probabilities.
- Interactive Streamlit web application.

## Emotions Supported

- Joy
- Sadness
- Anger
- Fear
- Love
- Surprise

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Matplotlib
- Streamlit

## Model Performance

| Metric | Score |
|---|---:|
| Accuracy | 90.43% |
| Macro F1 | 86.87% |
| Weighted F1 | 90.00% |
| Log Loss | 0.2835 |

*Results from the held-out test set.*

## Project Structure

```text
EmotiSense/
├── app/
│   └── app.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_model_comparison.ipynb
├── tests/
│   └── test_robustness.py
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/EmotiSense.git
cd EmotiSense
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app/app.py
```

## Future Improvements

- Improve handling of negation and spelling errors.
- Explore transformer-based NLP models.
- Improve performance on minority emotion classes.

## Disclaimer

EmotiSense is an experimental machine learning project. Predictions are estimates and should not be treated as psychological assessments.
