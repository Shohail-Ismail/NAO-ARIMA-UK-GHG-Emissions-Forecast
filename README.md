# UK GHG Emissions Forecast with ARIMA for National Audit Office (NAO)

   * [Overview](#overview)
   * [Core features](#core-features)
   * [Limitations and future developments](#limitations-and-future-developments)
   * [Environment setup](#environment-setup)

## Overview
A statistical analysis of the UK's greenhouse gas emissions from 1990 to 2023 using the Office for National Statistics' 'Atmospheric emissions' dataset, using `ARIMA` (Autoregressive Integrated Moving Average) for time series forecasting and `pmdarima.auto_arima` for automatic parameter selection to optimise the model based on the Akaike Information Criterion (AIC). This was done as part of an assignment given by the National Audit Office and includes data visualisation, exploratory analysis, stationarity tests, and and a 5-year forecast with confidence intervals. 

The statistical analysis involved transforming non-stationary data `(ADF p-value = 0.99)` into a stationary series `(ADF p-value = 0.00)`. Using `ARIMA(1,0,0)`, via `auto_arima` with `AIC = 748.86`, it forecasts emissions for 2024 - 2028 with 95% confidence intervals.

## Core features

-  **Exploratory Data Analysis (EDA)**: Computed key metrics such as mean, standard deviation, and quartiles to understand the dataset's properties, after which the raw historical GHG data was visualised to detect trends (identifying a clear downward trend without significant anomalies) and inform decision-making on the choice of model used. As the dataset was fairly small, all the values could be checked manually for integrity, which is why no preprocessing was needed other than format conversion.

-  **Stationarity testing**: Augmented Dickey-Fuller (ADF) Test was used to conduct stationarity testing, showing non-stationarity eithin the data with an ADF statistic of `0.79 (p-value = 0.99)`. To rectify this, applied differencing to stabilise the mean and remove non-stationarity, achieving an ADF statistic of `-7.65 (p-value = 0)`, thereby confirming stationarity. Although automated differencing exists in in `auto_arima`, opted for manual differencing due to more reliable confidence intervals.

-  **ARIMA modelling**: Ended with an optimal order of `ARIMA(1,0,0)` with an intercept term; the `AR(1)` coefficient `(0.9998)` indicates strong persistence, reflecting significant autocorrelation in the time series (as expected). Additionally, conducting Ljung-Box Q Test and Jarque-Bera tests confirmed that the model residuals lacked autocorrelation and were approximately normally distributed.

-  **5-year forecast**: Generated predictions for 2024 – 2028 using the fitted ARIMA model and plotted alongside historical data, incorporating 95% confidence intervals for uncertainty estimation.

## Limitations and future developments

One limitation of this model is its reliance on a small dataset, which restricts the quality of predictions and contributes to challenges during the parameterfitting stage, as reflected through the displayed warnings (i.e., near-singular covariance matrix warning and using outer product of gradients instead of Hessian method) . Additionally, there is a heavy reliance on historical trends and assumptions of linearity.

Further developments could include using a larger dataset or more granular (e.g., monthly or quarterly) data for more accurate results. Additionally, the impact of specific industries could be taken into consideration to inform better decision-making in policymaking and sector-specific legislation. While there is also an argument to incorporate external factors into the modelling to weigh in the impact of economic growth, global events, etc., this would require a machine learning model instead of simple statistical analysis.

## Environment setup

- **Required Python version**: Python 3.10.9
- **Libraries**:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `statsmodels`
  - `pmdarima`
  - `openpyxl`

-  **Structure**: Ensure the dataset `provisionalatmoshpericemissionsghg.xlsx` is in the root directory of your project.
 
- **Running the program**:

```bash
# To run the analysis
python NAO_Task.py
```
