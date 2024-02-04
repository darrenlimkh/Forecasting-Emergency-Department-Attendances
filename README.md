# Enhancing Emergency Department Attendance Predictions

## File Structure
```
+--- models
¦    {year_start}-{year_end}-{index}.pkl
¦    rf.pkl
¦    hyperparameters.pkl
¦    hyperparameters_dropped.pkl
+--- pickles
¦    weekly_slopes.pkl
¦    duration.pkl
¦    weekly_slopes_dropped.pkl
¦    duration_dropped.pkl
¦    metrics.csv
¦    forecasts.csv
+--- data.csv
¦    
+--- README.md
¦    
+--- submission.ipynb
```

## File Description
./models/{year_start}-{year_end}-{index}.pkl : Each pickle file corresponds to a trained SARIMAX model. year_start – year_end corresponds to the training period for which the SARIMAX model is trained on. index corresponds to the dataset reference, where 0 – 6 are block bootstrapped data and 7 is the original dataset.

./models/rf.pkl : Pre-trained trend prediction model

./models/hyperparameters.pkl : dictionary of hyperparameters that were tuned for the TPM. Keys are the models, values are the hyperparameter values

./models/hyperparameters_dropped.pkl : same as above, but models were tuned on a training set that does not include drifts that are close to an extrema

./pickles/weekly_slopes.pkl : A nested dictionary with the following structure {index : { (timestamp, count) : [slope1, slope2, …, slope7], … }}, where index corresponds to the bootstrapped dataset reference. timestamp corresponds to the drift timestamp, and the count denotes the time instance where the slopes were computed (e.g., 1 implies that the slope is computed when the drift is detected, 2 implies that the slope is computed in the following week). The predictors for the TPM are extracted from this file. 

./pickles/duration.pkl : A nested dictionary with the following structure {index : { timestamp : (slope, duration), … }}, where index corresponds to the bootstrapped dataset reference. timestamp corresponds to the drift timestamp, and the duration is the period until the next turning point. The target for the TPM is extracted from this pickle file. 

./pickles/weekly_slopes_dropped.pkl : same as above, but models were tuned on a training set that does not include drifts that are close to an extrema

./pickles/duration_dropped.pkl : same as above, but models were tuned on a training set that does not include drifts that are close to an extrema

./pickles/metrics.csv : A dataframe of the performance metrics (i.e., MAE, MAPE, MSE, RMSE) of the SARIMAX models in each year (note that it is based on a yearly retraining and incremental learning within the year)

./pickles/forecasts.csv : A dataframe of daily actual, forecasts, and errors. The timestamp spans from 2014 to 2022.

./data.csv : A dataframe of daily actual cases (raw dataset). The timestamp spans from 2010 to early 2023. 

./README.md : existing documentation file

./submission.ipynb : source code
