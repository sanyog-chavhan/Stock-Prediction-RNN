# Stock Price Prediction with a Two-Layer RNN

A PyTorch replication of an RNN-based stock price prediction experiment, applied to Apple (AAPL) daily Open prices. The model uses the last 5 days of prices to predict the next day's price.

**Paper:** [Stock price prediction using the RNN model](https://iopscience.iop.org/article/10.1088/1742-6596/1650/3/032103), Yongqiong Zhu, *Journal of Physics: Conference Series*, Volume 1650, 2020. DOI: [10.1088/1742-6596/1650/3/032103](https://doi.org/10.1088/1742-6596/1650/3/032103)

## Files

| File | Purpose |
|---|---|
| `fetch_aapl.py` | Downloads AAPL price history with `yfinance` and saves it to `aapl_prices.csv` |
| `aapl_prices.csv` | Daily AAPL prices from 2009-08-10 onwards (the data used for the experiment) |
| `rnn_pytorch_stock_prediction.py` | Data preparation, model, training, validation, test evaluation and plots |

## Setup

Requires Python 3 with the following packages:

```
pip install torch pandas numpy scikit-learn matplotlib yfinance
```

`yfinance` is only needed if you want to re-download the data.

## Running the experiment

```
python rnn_pytorch_stock_prediction.py
```

The script trains on the first 65% of the data and tests on the rest. It prints the loss for each epoch, the test MSE, RMSE and MAE in dollars, and plots the training curves and the actual vs predicted test prices.

To re-download the data first, run `python fetch_aapl.py`. This overwrites `aapl_prices.csv`.

## Model

| Setting | Value |
|---|---|
| Input size | 1 (scaled Open price) |
| Timesteps | 5 |
| First RNN layer | 50 units |
| Second RNN layer | 100 units |
| Dropout | 0.2, between the two RNN layers |
| Output | 1 (next day's scaled Open price) through a linear layer |
| Batch size | 64 |
| Epochs | 50 |
| Optimiser | Adam (default learning rate 0.001) |
| Loss | MSE |

## Data handling

- The first 65% of days are used for training and the last 35% for testing.
- The last 10% of the training windows are held out as a validation set, in time order.
- Prices are scaled to [0, 1] with `MinMaxScaler`, fit on the training data only.
- The training loader shuffles each epoch. The validation and test loaders keep time order.
- Random seeds are fixed (`SEED = 42`) so runs are repeatable.

Choices not given in the paper, such as the 65/35 split, the validation split, the learning rate and the random seed, were made here and may differ from the original experiment.

## Results

Test set metrics, with predictions converted back to USD:

| Metric | Timesteps = 5 | Timesteps = 10 |
|---|---|---|
| MSE (USD²) | 82.8626 | 114.7561 |
| RMSE (USD) | 9.1029 | 10.7124 |
| MAE (USD) | 5.3776 | 7.3850 |

The model's predictions sit consistently below the actual price, and the gap widens through 2020. The paper reports a prediction accuracy of over 95% and a loss close to 0.1%. These results don't reproduce that. The paper's accuracy measure and data split aren't described on its abstract page, so the two can't be compared directly.
