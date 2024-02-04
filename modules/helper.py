import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.svm import SVR
import xgboost as xgb


def outlierLimits(array):
  """ returns the lower and upper limit of outlier threshold (75th + 1.5IQR / 25th - 1.5IQR)
  :param array: an array of numbers
  :type array: np.array
  ...
  :return: lower and upper limits
  :rtype: float
  """
  percentile_75 = np.percentile(array, 75)
  percentile_25 = np.percentile(array, 25)
  iqr = percentile_75 - percentile_25
  upper = np.round(percentile_75 + 1.5*iqr, 0)
  lower = np.round(percentile_25 - 1.5*iqr, 0)
  return lower, upper

def checkThreshold(series, upper_limit, lower_limit):
  """ returns the number of elements that are above/below defined limits
  :param series: an array of numbers
  :type series: np.array
  :param upper_limit: upper limit
  :type upper_limit: float
  :param lower_limit: lower limit
  :type lower_limit: float
  ...
  :return: number of elements that are outside given limits
  :rtype: int
  """
  return sum(series > upper_limit) + sum(series < lower_limit)

def checkThreshold2(series, upper_limit, lower_limit, window, condition, upper=True):
  """ returns the elements that are above/below defined limits across a moving window. E.g., condition = 4 and window = 5
  implies that if 4 out of 5 points are outside given limits, extract those relevant timestamps
  :param series: an array of numbers
  :type series: np.array
  :param upper_limit: upper limit
  :type upper_limit: float
  :param lower_limit: lower limit
  :type lower_limit: float
  :param window: window size
  :type window: int
  :param condition: minimum number of points to be considered as an out-of-control
  :type condition: int
  :param upper: to check against upper limit, else check against lower limit
  :type upper: bool
  ...
  :return: timestamps which satisfy out-of-control conditions
  :rtype: list
  """
  continuous_timestamps = []
  start = window - 1
  for index in range(start, len(series)):
    subset = series.iloc[index-start: index+1]
    if upper:
      index = subset[subset > upper_limit].index.tolist()
    else:
      index = subset[subset < lower_limit].index.tolist()
    if len(index) < condition:
      continue
    else:
      continuous_timestamps.extend(index)
  continuous_timestamps = sorted(list(set(continuous_timestamps)))
  return continuous_timestamps

def controlChart(train_errors, test_errors, num_exceed=1, ewma_factor=0.3, verbose=True):
  """ determines whether a set of new errors is within/outside controls
  :param train_errors: historical data to build the control chart
  :type train_errors: pd.Series
  :param test_errors: existing forecat errors to check against for control
  :type test_errors: pd.Series
  :param num_exceed: threshold to determine whether to flag out-of-control
  :type num_exceed: int
  :param ewma_factor: smoothing paramter
  :type ewma_factor: float
  :verbose: prints timestamps that exceed control limits into the console & plot the control chart
  :type verbose: bool
  ...
  :return: "Out-of-Control"/"Within Control" and timestamps which satisfy out-of-control condition
  :rtype: string, list
  """
  center_line = train_errors.mean()
  std_dev = (train_errors.var() * (ewma_factor / (2 - ewma_factor))) ** 0.5
  control_limit_upper = center_line + 3*std_dev
  control_limit_lower = center_line - 3*std_dev
  errors = pd.concat([pd.Series(center_line), test_errors], axis=0)
  ewma = pd.Series(errors).ewm(alpha=ewma_factor).mean().iloc[1:]
  ewma_eval = ewma[-7:]
  if checkThreshold(ewma_eval, control_limit_upper, control_limit_lower) >= num_exceed:
    upper_mask = ewma_eval.gt(control_limit_upper)
    upper_indices = upper_mask.index[upper_mask].tolist()
    lower_mask = ewma_eval.lt(control_limit_lower)
    lower_indices = lower_mask.index[lower_mask].tolist()
    timestamps = sorted(lower_indices + upper_indices)
    drift_points = ewma_eval.loc[timestamps]
    timestamp_strings = [timestamp.strftime('%Y-%m-%d') for timestamp in timestamps]
    if verbose:
      if len(timestamp_strings) > 1:
        print(f'UPDATE {timestamp_strings[0]} - {timestamp_strings[-1]}: error exceeds control limits')
      else:
        print(f'UPDATE {timestamp_strings[0]}: error exceeds control limits')
      plt.figure(figsize=(10, 5))
      plt.plot(ewma.index, ewma.values, label='EWMA', color='blue')
      plt.scatter(errors.iloc[1:].index, errors.iloc[1:].values, label='Errors', color='black')
      plt.scatter(drift_points.index, drift_points.values, label='Out-of-Control Points', color='red')
      plt.axhline(center_line, color='r', linestyle='--', label='Center Line')
      plt.axhline(control_limit_upper, color='g', linestyle='--', label='Upper Control Limit')
      plt.axhline(control_limit_lower, color='g', linestyle='--', label='Lower Control Limit')
      plt.xlabel('Time')
      plt.ylabel('EWMA')
      plt.title('EWMA Control Chart')
      plt.legend()
      plt.show();
    return 'Out of Control!', timestamps
  else:
    return 'Within Control!', []

def controlChart2(train_errors, test_errors, ewma_factor=0.3, window=7, condition=4, verbose=True):
  """ determines whether a set of new errors is within/outside controls. This second function provides more flexibility in terms of allowing a X number
  of points within a Y window to be ascertained as an out-of-control condition
  :param train_errors: historical data to build the control chart
  :type train_errors: pd.Series
  :param test_errors: existing forecat errors to check against for control
  :type test_errors: pd.Series
  :param num_exceed: threshold to determine whether to flag out-of-control
  :type num_exceed: int
  :param ewma_factor: smoothing paramter
  :type ewma_factor: float
  :verbose: prints timestamps that exceed control limits into the console & plot the control chart
  :type verbose: bool
  ...
  :return: "Out-of-Control"/"Within Control" and timestamps which satisfy out-of-control condition
  :rtype: string, list
  """
  center_line = train_errors.mean()
  std_dev = (train_errors.var() * (ewma_factor / (2 - ewma_factor))) ** 0.5
  control_limit_upper_2 = center_line + 2*std_dev
  control_limit_lower_2 = center_line - 2*std_dev
  control_limit_upper_3 = center_line + 3*std_dev
  control_limit_lower_3 = center_line - 3*std_dev
  errors = pd.concat([pd.Series(center_line), test_errors], axis=0)
  ewma = pd.Series(errors).ewm(alpha=ewma_factor).mean().iloc[1:]
  ewma_eval = ewma[-7:]
  exceed_upper = checkThreshold2(ewma_eval, control_limit_upper_2, control_limit_lower_2, window=window, condition=condition, upper=True)
  exceed_lower = checkThreshold2(ewma_eval, control_limit_upper_2, control_limit_lower_2, window=window, condition=condition, upper=False)
  if exceed_lower or exceed_upper:
    timestamps = sorted(exceed_lower + exceed_upper)
    drift_points = ewma_eval.loc[timestamps]
    timestamp_strings = [timestamp.strftime('%Y-%m-%d') for timestamp in timestamps]
    if verbose:
      if len(timestamp_strings) > 1:
        print(f'{timestamp_strings[0]} - {timestamp_strings[-1]}: error exceeds control limits')
      else:
        print(f'{timestamp_strings[0]}" error exceeds control limits')
      plt.figure(figsize=(10, 5))
      plt.plot(ewma.index, ewma.values, label='EWMA', color='black')
      plt.scatter(errors.iloc[1:].index, errors.iloc[1:].values, label='Errors', color='black')
      plt.scatter(drift_points.index, drift_points.values, label='Out-of-Control Points', color='red')
      plt.axhline(center_line, color='black', linestyle='--', label='Center Line')
      plt.axhline(control_limit_upper_2, color='b', linestyle='--', label='Upper Control Limit (2SD)')
      plt.axhline(control_limit_lower_2, color='b', linestyle='--', label='Lower Control Limit (2SD)')
      plt.axhline(control_limit_upper_3, color='r', linestyle='--', label='Upper Control Limit (3SD)')
      plt.axhline(control_limit_lower_3, color='r', linestyle='--', label='Lower Control Limit (3SD)')
      plt.xlabel('Time')
      plt.ylabel('EWMA')
      plt.title('EWMA Control Chart')
      plt.legend()
      plt.show();
    return 'Out of Control!', timestamps
  else:
    return 'Within Control!', []

def getMedianTimestamp(timestamps):
  """ returns the median value given a list (used to extract the median timestamp)
  :param timestamps: an array of timestamps
  :type timestamps: np.array
  ...
  :return: median timestamp
  :rtype: pd.TimeStamp
  """
  num_timestamps = len(timestamps)
  median_index = num_timestamps // 2
  if num_timestamps % 2 == 1:
    median = timestamps[median_index]
  else:
    median = timestamps[median_index-1]
  return median

def indexPair(numbers):
  """ assigns an index to each element (used to enumerate kernel-regressed values)
  :param numbers: an array of numbers
  :type timestamps: np.array
  ...
  :return: the enumerated array of numbers
  :rtype: np.array
  """
  indices = np.arange(1, len(numbers) + 1)
  result = np.column_stack((numbers, indices))
  return result

def estimateSlope(lowess_curve, idx):
  """ computes slope using local linear regression
  :param lowess_curve: an array of numbers pertaining to the smoothed number of cases using LOWESS or Kernel Regression
  :type lowess_curve: np.array
  :param idx: index along the array of interest to compute the slope
  :type idx: int
  ...
  :return: slope value
  :rtype: float
  """
  y_neigh = np.array(lowess_curve[idx-1:idx+2, 0])
  x_neigh = np.array(lowess_curve[idx-1:idx+2, 1]).reshape(-1, 1)
  model = LinearRegression(fit_intercept=True).fit(x_neigh, y_neigh)
  slope = model.coef_[0]
  return slope

def plotWarnings(index, actual, lowess, median_timestamps, extrema):
  """ displays the plot of smoothed curve, demarcate turning points and drift detections
  :param index: an array of timestamps
  :type index: np.array
  :param actual: an array of targets (cases)
  :type actual: np.array
  :param lowess: an array of smoothed values of targets (cases)
  :type lowess: np.array
  :param median_timestamps: an array of median timestamps where drift was detected
  :type median_timestamps: np.array
  :param extrema: an array of index to denote turning points
  :type extrema: np.array
  ...
  :return: does not return any value
  :rtype: None
  """
  plt.figure(figsize=(10, 5))
  # plt.xlim(pd.Timestamp('2020-01-01'), pd.Timestamp('2022-12-31'))
  plt.scatter(index, actual, color='black', label='Actual')
  plt.plot(index, lowess, color='blue', label='Fitted')
  for median_timestamp in median_timestamps:
    plt.axvline(x=median_timestamp, color='green', linestyle='--')
  for point in extrema:
    plt.scatter(index[point], lowess[point], marker='x', color='red')
  plt.title('Cases against Time')
  plt.xlabel('Time')
  plt.ylabel('Cases')
  plt.legend()
  plt.grid(False)
  plt.show();

def plotWarnings2(index, actual, lowess, median_timestamps, extrema):
  """ displays the plot of smoothed curve, demarcate turning points and drift detections
  :param index: an array of timestamps
  :type index: np.array
  :param actual: an array of targets (cases)
  :type actual: np.array
  :param lowess: an array of smoothed values of targets (cases)
  :type lowess: np.array
  :param median_timestamps: an array of median timestamps where drift was detected
  :type median_timestamps: np.array
  :param extrema: an array of index to denote turning points
  :type extrema: np.array
  ...
  :return: does not return any value
  :rtype: None
  """
  plt.figure(figsize=(10, 5))
  plt.xlim(pd.Timestamp('2020-01-01'), pd.Timestamp('2022-12-31'))
  plt.plot(index, lowess, color='blue', label='Kernel Regressed')
  for point in extrema:
    plt.scatter(index[point], lowess[point], marker='x', color='red')
  for median_timestamp in median_timestamps:
    plt.axvline(x=median_timestamp, color='green', linestyle='--')

  plt.title('Cases against Time')
  plt.xlabel('Time')
  plt.ylabel('Cases')
  plt.legend()
  plt.grid(False)
  plt.show();

def findMinimaMaxima(series, neighborhood_width=30):
  """ returns the maxima and minima points in an array of numbers
  :param series: an array of values
  :type series: np.array
  :param neighbourhood_width: window to determine maximum/minimum
  :type neighbhourhood_width: int
  ...
  :return: an array of minima, an array of maxima
  :rtype: list, list
  """
  local_minima = []; local_maxima = []
  for i in range(neighborhood_width, len(series) - neighborhood_width):
      neighbors = series[i-neighborhood_width:i+neighborhood_width+1]
      if series[i] == np.min(neighbors):
          local_minima.append(i)
      elif series[i] == np.max(neighbors):
          local_maxima.append(i)
  return local_minima, local_maxima

def findLargerNumber(arr, target):
  """ finds the closest larger number to the target given an array of numbers. This is used to identify the extrema point after a drift timestamp
  :param arr: an array of values (extrema values)
  :type arr: np.array
  :param target: target index (drift index)
  :type target: int
  ...
  :return: the extrema index
  :rtype: int
  """
  closest_larger = float('inf')
  for num in arr:
    if num > target and num < closest_larger:
      closest_larger = num
  return closest_larger

def extractValues(dictionary, name):
  """ given a dictionary with keys (timestamp, index), extract values of the same timestamp
  :param dictionary: a dictionary with mentioned key structure
  :type dicitonary: dict
  :param name: timestamp
  :type name: pd.TimeStamp
  ...
  :return: values associated with timestamp
  :rtype: list
  """
  values = []
  for key in dictionary:
    if key[0] == name:
      values.append(dictionary[key])
  return values

def plotPredictions(summary_forecasts):
  """ plots yearly predictions against actual cases
  :param summary_forecasts: forecasted values
  :type summary_forecasts: pd.DataFrame
  ...
  :return: does not return any value
  :rtype: None
  """
  years = summary_forecasts.index.year.unique()
  for year in years:
    summary_forecasts_year = summary_forecasts[summary_forecasts.index.year == year]
    plt.figure(figsize=(15,6))
    plt.plot(summary_forecasts_year.index, summary_forecasts_year['Actual'], label='Actual', color='black')
    plt.plot(summary_forecasts_year.index, summary_forecasts_year['Forecast'], label='Prediction', color='red')
    plt.xlabel('Date')
    plt.ylabel('Cases')
    plt.legend()
    plt.show();

def hasNestedList(lst):
  """ check if a list is nested
  :param lst: a list of values
  :type lst: list
  ...
  :return: True/False indicator
  :rtype: boolean
  """
  for element in lst:
    if isinstance(element, list):
      return True
  return False

def createDataset(syn_slopes, syn_results, timestamp_start, timestamp_end, threshold=60, median=False):
  """ generates slope-duration pair for subsequent machine learning algorithms
  :param syn_slopes: drift_timestamp and their corresponding slope values
  :type syn_slopes: dictionary
  :param syn_results: drift_timestamp and their corresponding duration values
  :type syn_results: dictionary
  :param timestamp_start: time constraint to denote section of data that is collated for training
  :type timestamp_start: int
  :param timestamp_end: time constraint to denote section of data that is collated for training
  :type timestamp_end: int
  :param threshold: threshold to remove any outliers
  :type thershold: int
  :param median: indicator to store only the median slope or all the slopes for training
  :type median: bool
  ...
  :return: slope, duration pairs
  :rtype: list, list
  """
  ref = 15; train = {}; all_data = []; x = []; y = []

  ## Extract only the slope of interest, with respect to the lagged reference value
  for index, info in syn_slopes.items():
    sample = {}
    for timestamp_index, values in info.items():
      if timestamp_index[-1] == ref and (timestamp_start <= timestamp_index[0].year <= timestamp_end):
        try: # point dont exist because its taken out due to near turning point
          sample[timestamp_index[0]] = (values, syn_results[index][timestamp_index[0]][-1])
        except:
          continue
    train[index] = sample

  for index, values in train.items():
    for timestamp, data in values.items():
      all_data.append(data)

  for index in range(len(all_data)):
    item = all_data[index]
    slope = item[0]; days = item[1]
    if days >= threshold:
      continue
    if median:
      x.append(slope[3])
    else:
      x.append(slope)
    y.append(days)

  if any(isinstance(element, list) for element in x):
    X = np.array(x)
  else:
    X = np.array(x).reshape(-1, 1)

  return X, y

def createTest(test_syn_slopes, timestamp_start, timestamp_end, delay, median=False):
  """ generates slope-duration pair for subsequent machine learning algorithms
  :param test_syn_slopes: drift_timestamp and their corresponding slope values
  :type test_syn_slopes: dictionary
  :param timestamp_start: time constraint to denote section of data that is collated for testing
  :type timestamp_start: int
  :param timestamp_end: time constraint to denote section of data that is collated for testing
  :type timestamp_end: int
  :param delay: store slope values after X weeks of delay
  :type delay: int
  :param median: indicator to store only the median slope or all the slopes for testing
  :type median: bool
  ...
  :return: testing set
  :rtype: np.array
  """
  test = {}
  for timestamp_index, values in test_syn_slopes.items():
    # if timestamp_index[-1] == delay and timestamp_index[0] in syn_results[7].keys():
    if timestamp_index[-1] == delay:
      if (timestamp_start <= timestamp_index[0].year <= timestamp_end):
        if median:
          test[timestamp_index[0]] = values[3]
        else:
          test[timestamp_index[0]] = values
  if median:
    X_test = np.array(list(test.values())).reshape(-1, 1)
  else:
    X_test = np.array(list(test.values()))
  return X_test

def createActual(test_syn_results, timestamp_start, timestamp_end):
  """ generates actual duration for performance metric computations
  :param test_syn_results: drift_timestamp and their corresponding slope values
  :type test_syn_resultss: dictionary
  :param timestamp_start: time constraint to denote section of data that is collated for testing
  :type timestamp_start: int
  :param timestamp_end: time constraint to denote section of data that is collated for testing
  :type timestamp_end: int
  ...
  :return: actual values
  :rtype: list
  """
  actual = []
  for timestamp_index, values in test_syn_results.items():
    if (timestamp_start <= timestamp_index.year <= timestamp_end):
      actual.append(values[-1])
  return actual

def jackKnife(hyperparameters, X_train, y_train, X_test, y_test, folds=10, alpha=0.05):
  """ computes the jackknife estimate of the MAE mean, variance and 95% CI
  :param hyperparameters: set of hyperparameters pertaining to each model
  :type hyperparameters: dict
  :param X_train: train predictors
  :type X_train: np.array
  :param y_train: train target
  :type y_train: np.array
  :param X_test: test predictors
  :type X_test: np.array
  :param y_test: test target
  :type y_test: np.array
  :param fold: K-fold, sample size of jackknife estimate
  :type fold: int
  :param alpha: significance level
  :type alpha: float
  ...
  :return: dataframe of jackknife MAE mean, std, 95% lower and upper bound for each ML model
  :rtype: pd.DataFrame
  """
  mapping = {1: 'knn', 2:'et', 3:'rf', 4:'gbr', 5:'xgboost', 6:'dt', 7:'svm', 8:'lr'}

  kf = KFold(n_splits=folds, shuffle=True, random_state=42)
  jk_df = pd.DataFrame(columns=['model', 'mean', 'std', 'lower_bound', 'upper_bound'])
  res = []; estimators = {}
  for train_index, test_index in kf.split(X_train):
    X_train_, X_test_ = X_train[train_index], X_train[test_index]
    y_train_, y_test_ = np.array(y_train)[train_index], np.array(y_train)[test_index]

    est1 = KNeighborsRegressor(**hyperparameters['knn'])
    est2 = ExtraTreesRegressor(**hyperparameters['et'])
    est3 = RandomForestRegressor(**hyperparameters['rf'])
    est4 = GradientBoostingRegressor(**hyperparameters['gbr'])
    est5 = xgb.XGBRegressor(**hyperparameters['xgboost'])
    est6 = DecisionTreeRegressor(**hyperparameters['dt'])
    est7 = SVR(**hyperparameters['svm'])
    # est8 = LinearRegression(**hyperparameters['lr'])
    ests = [est1, est2, est3, est4, est5, est6, est7]

    for est_index in range(1, len(ests)+1):
      model = ests[est_index-1].fit(X_train_, y_train_)
      if est_index not in estimators:
        estimators[est_index] = [model]
      else:
        estimators[est_index].append(model)

  for est_index in estimators.keys():
    y_pred_multi = np.column_stack([e.predict(X_test) for e in estimators[est_index]])
    results = pd.DataFrame(y_pred_multi, columns=[f'model_{i}' for i in range(1, len(estimators[est_index])+1)])
    # mae_values = np.array(results.apply(lambda col: mae(col, y_test)))
    mae_values = np.abs(results.values - np.array(y_test).reshape(-1, 1)).mean(axis=0)

    n = len(mae_values)
    sample_mean = np.mean(mae_values)

    jackknife_means = np.zeros(n)
    for i in range(n):
      jackknife_data = np.delete(mae_values, i)
      jackknife_means[i] = np.mean(jackknife_data)

    jackknife_se = np.sqrt((n - 1) * np.var(jackknife_means, ddof=1))
    bias_correction = (n - 1) * (sample_mean - jackknife_means)

    jackknife_corrected_means = sample_mean - bias_correction
    jacknife_mean = np.mean(jackknife_corrected_means)
    z_critical = np.abs(np.percentile(jackknife_corrected_means, alpha / 2 * 100))

    lower_bound = jacknife_mean - z_critical * jackknife_se
    upper_bound = jacknife_mean + z_critical * jackknife_se

    jk_df = jk_df.append({'model': mapping[est_index], 'mean': jacknife_mean, 'std': jackknife_se, 'lower_bound': lower_bound, 'upper_bound': upper_bound}, ignore_index=True)

  return jk_df

def plot_confidence_interval(x, mean, bottom, top, z=1.96, color='#2187bb', horizontal_line_width=0.25):
  """ visualize the jackknife 95% CIs
  :param x: index
  :type x: int
  :param bottom: lower bound
  :type bottom: float
  :param top: upper bound
  :type top: float
  ...
  :return: does not return any value
  :rtype: None
  """
  left = x - horizontal_line_width / 2
  right = x + horizontal_line_width / 2
  plt.plot([top, bottom], [x, x], color=color)
  plt.plot([top, top], [left, right], color=color)
  plt.plot([bottom, bottom], [left, right], color=color)
  plt.plot(mean, x, 'o', color='#f44336')