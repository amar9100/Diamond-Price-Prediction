# 💎 Diamond Price Prediction Using KNN and Streamlit

## Project Overview
This project predicts diamond prices using a K-Nearest Neighbors (KNN) regression model. A Streamlit web application allows users to enter diamond characteristics and receive an estimated price.

## Objectives
- Explore and preprocess the diamond dataset.
- Train a KNN regression model.
- Evaluate model performance using regression metrics.
- Deploy an interactive prediction interface using Streamlit.

## Dataset
The project uses `diamonds.csv`, which contains diamond characteristics such as carat, cut, color, clarity, depth, table, dimensions, and price.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Pickle

## Machine Learning Workflow
1. Load and explore the dataset.
2. Separate features and target variable.
3. Split data into training and testing sets.
4. Scale numerical features using StandardScaler.
5. Encode categorical features using OneHotEncoder.
6. Train a KNN regression model.
7. Evaluate model performance.
8. Save the trained pipeline.
9. Use Streamlit to predict diamond prices.

## Model Performance

| Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | 408.17 |
| Root Mean Squared Error (RMSE) | 785.87 |
| R² Score | 0.9607 |

## Project Files
- `diamonds.csv` — Dataset
- `description.txt` — Dataset description
- `train_model.py` — Model training and evaluation
- `diamond_model.pkl` — Saved trained model
- `app.py` — Streamlit web application
- `README.md` — Project documentation

## Installation

Install the required Python libraries:

```bash
pip install pandas numpy scikit-learn streamlit
```

## How to Run

### Step 1: Train the model
```bash
python train_model.py
```

### Step 2: Start the Streamlit application
```bash
streamlit run app.py
```

The application will open in your web browser.

## Prediction
Enter the diamond's characteristics and click **Predict Diamond Price** to see the model's estimated price.

## Conclusion
The KNN regression model achieved an R² score of 0.9607 on the test dataset. The trained pipeline is integrated into a Streamlit application that provides interactive diamond price predictions.

The predicted price is an estimate and may differ from actual market prices.