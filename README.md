# Predictive Analytics Using Historical Data

## Project Overview
This project builds a machine-learning model to forecast future monthly sales trends from historical data.

### Objectives
- Clean and preprocess historical data.
- Analyze sales trends.
- Engineer time-series features.
- Train a regression model.
- Evaluate prediction accuracy.
- Forecast the next 6 months.
- Visualize actual vs predicted and future values.

## Technologies
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## Dataset
The project contains monthly sales data from January 2018 to December 2025 with:
- Date
- Sales
- Promotion indicator

The dataset includes a small number of missing sales values to demonstrate preprocessing.

## Methodology
1. Load historical sales data.
2. Sort records chronologically.
3. Handle missing sales using interpolation.
4. Create year, month, and time-index features.
5. Create lag features for 1, 3, 6, and 12 months.
6. Create 3-month and 6-month rolling averages.
7. Split data chronologically into 80% training and 20% testing data.
8. Train a Random Forest Regression model.
9. Evaluate using MAE, RMSE, R² and MAPE.
10. Generate a six-month recursive forecast.

## Model Evaluation
The generated project evaluates the model using:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- Mean Absolute Percentage Error (MAPE)

## Future Forecast
The project produces a six-month forecast. For this demonstration, future promotion status is assumed to be 0. In a real business scenario, future promotional plans should be supplied as model inputs.

## Business Applications
Predictive sales analytics can support:
- Inventory planning
- Sales target setting
- Marketing planning
- Demand forecasting
- Resource allocation
- Trend monitoring

## Files
- `Predictive_Analytics_Project.ipynb` — complete notebook
- `predictive_analytics.py` — Python implementation
- `historical_sales_data.csv` — historical dataset
- `future_sales_forecast.csv` — six-month predictions
- `model_metrics.csv` — evaluation metrics
- PNG files — visualizations

## How to Run
```bash
pip install pandas numpy matplotlib scikit-learn jupyter
jupyter notebook Predictive_Analytics_Project.ipynb
```

**Note:** The included dataset is synthetic and created for educational/demo purposes.
