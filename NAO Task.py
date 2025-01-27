import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller

# Deprecation warning suppressed for output clarity
warnings.simplefilter(action = 'ignore', category = FutureWarning)

# Cleans and preps data
def preprocess_data():
    
    # Skips unnecessary info by only using 2 columns from 'GHG total' sheet
    df = pd.read_excel(
        "provisionalatmoshpericemissionsghg.xlsx", 
        sheet_name = 'GHG total', skiprows = 7, nrows = 34, usecols = "A, Y"
    )

    # Rename columns for ease of use
    df.columns = ["Year", "GHG_Total"]

    # Converts data to correct formats
    df["GHG_Total"] = pd.to_numeric(df["GHG_Total"])
    df["Year"] = pd.to_datetime(df["Year"].astype(int), format ="%Y") + pd.offsets.YearEnd(0)

    # Converts dataframe into time-series data and prepares it
    ghg_ts = pd.Series(df["GHG_Total"].values, index = df["Year"])
    ghg_ts = ghg_ts.astype(float)
    ghg_ts.index = pd.DatetimeIndex(ghg_ts.index, freq = "YE")
    
    # Returns time series and cleaned dataframe
    return ghg_ts, df

# Simple exploratory data analysis to gauge basic properties
def eda(df):
    print("Explaratory Data Analysis: ")
    print(df.describe())
    
    # Plots graph of data to gauge visual trend(s)
    plt.figure(figsize = (8, 5))
    plt.plot(df['Year'], df['GHG_Total'], marker = 'x')
    plt.title('Raw data: 1990-2023 UK GHG emissions ')
    plt.xlabel('Year (end of)')
    plt.ylabel('GHG emissions (kt of CO2 equivalent)')
    plt.grid(True)
    plt.show()

# Automatically determines optimal (p,d,q) params for ARIMA 
def auto_arima_order(ghg_ts):
    
    # Along with std params, max AR and MA terms to test set to 3 due to small data
    # Random seed of 42 also set for reproducibility in report
    auto_model = auto_arima(
        ghg_ts, 
        seasonal = False,
        trace = True,
        error_action = 'ignore',
        suppress_warnings = True,
        stepwise = True,
        max_p = 3,
        max_q = 3,
        random_state = 42
    )
    return auto_model.order

# Fits model to pre-selected order
def fit_arima_model(ghg_ts, order):
    # Linear trend chosen as similar trend seen visually during  EDA
    # Even if trend in EDA was quadratic, differencing will make it linear
    model = ARIMA(ghg_ts, order = order, trend = 't')
    results = model.fit()
    return results

# Forecasts GHG emissions for next 'steps' years
def predict_GHG(model_fit, steps = 5):
    
    # Fits model
    forecast_result = model_fit.get_forecast(steps)
    
    # Returns means and confidence intervals 
    forecast_df = forecast_result.summary_frame()
    return forecast_df

# Plots historical data alongside forecasted values
def plot_forecast(ghg_ts, forecast_df):
    
    plt.figure(figsize = (10, 6))
    
    # Builds forecast years
    last_hist_year = ghg_ts.index.year[-1]
    forecast_years = np.arange(last_hist_year + 1, last_hist_year + 1 + len(forecast_df))

    # Combines for a single 'historical + forecast' line
    combined_years = np.concatenate([ghg_ts.index.year, forecast_years])
    combined_vals = np.concatenate([ghg_ts.values, forecast_df["mean"].values])
    
    # Plot combined line
    plt.plot(
        combined_years, 
        combined_vals, 
        label = "Historical (1990 - 2023)", 
        marker = "x", 
        color = "blue"
    )
    
    # Highlight the forecasted portion in red
    plt.plot(
        forecast_years,
        forecast_df['mean'],
        label = "Forecast (2024 - 2028)", 
        marker = 'x',
        color = 'red'
    )
    
    # Confidence intervals
    plt.fill_between(
        forecast_years,
        forecast_df['mean_ci_lower'],
        forecast_df['mean_ci_upper'],
        color = 'pink',
        alpha = 0.3,
        label = '95% CI'
    )
    
    plt.title('UK GHG emissions, historical + forecast (2024 - 2028)')
    plt.xlabel('Year (end of)')
    plt.ylabel('GHG emissions (kt CO2 eq)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Perform Augmented Dicky-Fuller for stationarity in data (necessary for using ARIMA)
def check_stationarity(ghg_ts):
    print("\n########## ADF Test ############")
    result = adfuller(ghg_ts)
    print(f"ADF stat: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    print("Critical values: ")
    for key, value in result[4].items():
        print(f"   {key}: {value:.4f}")
    if result[1] < 0.05:
        print("Stationary: (p-value < 0.05)")
    else:
        print("Non-stationary (p-value >= 0.05) ")

# Applies first-order differencing to make data stationary
def difference_series(ghg_ts):
    differenced = ghg_ts.diff().dropna()

# Appears as white noise indicating no significant structure remains in data
# Not useful beyond informing order of differencing so commented out
"""
    # Plot the differenced series
    plt.figure(figsize = (8, 5))
    plt.plot(differenced, marker = 'x')
    plt.title('Differenced GHG Emissions Time Series')
    plt.xlabel('Year (end of)')
    plt.ylabel('Differenced GHG emissions (kt CO2 eq)')
    plt.grid(True)
    plt.show()
"""
    
    return differenced

def main():    
    # Prepare data as a time series
    ghg_ts, df_cleaned = preprocess_data()
    
    # Exploratory Data Analysis
    eda(df_cleaned)
    
    # Check stationarity of original series
    print("Checking stationarity of the original series:")
    check_stationarity(ghg_ts)
    
    # Apply differencing to make the series stationary (printed explicitly for demonstration)
    print("\nApplying first-order differencing...")
    differenced_ts = difference_series(ghg_ts)
    
    # Confirm stationarity of series
    print("Checking stationarity of the differenced series:")
    check_stationarity(differenced_ts)
    
    # Auto-select ARIMA order on differenced data
    order = auto_arima_order(differenced_ts)
    
    # Fit ARIMA model using that order on original series
    arima_model = fit_arima_model(ghg_ts, order)
    print("\nARIMA Model Summary")
    print(arima_model.summary())
        
    # Forecasts next 5 years
    forecast_df = predict_GHG(arima_model, 5)
    print(f"\n5-year forecast")
    print(forecast_df)
    
    # Plots forecast
    plot_forecast(ghg_ts, forecast_df)

if __name__ == "__main__":
    main()
