# Sales Prediction

Forecasting monthly sales per shop–item pair from daily historical sales data of a Russian retail chain.

## Problem

Given daily sales records from January 2013 to October 2015 (34 months), predict the total number of units sold for each (shop, item) pair in the following month.

## Data

Place the four CSV files in the `data/` folder

| File | Description |
|---|---|
| `sales_train.csv` | Daily sales: `date`, `date_block_num`, `shop_id`, `item_id`, `item_price`, `item_cnt_day` |
| `items.csv` | `item_id`, `item_name`, `item_category_id` |
| `shops.csv` | `shop_id`, `shop_name` |
| `item_categories.csv` | `item_category_id`, `item_category_name` |

Key columns:

- `date_block_num` is a consecutive month index (0 = January 2013, 33 = October 2015).
- `item_cnt_day` is the number of units sold per day; its monthly sum per (shop, item) is the **target**.
- `item_price` is the current price of an item.

Size: 2,935,849 daily records, 60 shops, 22,170 items (21,807 of them sold), 84 categories.

## Project structure

```
sales_prediction/
├── data/                        # raw CSV files and features.parquet (git-ignored)
├── notebooks/                   # EDA, DQC & ETL
├── reports/                     # model selection results (experiment output)
├── src/sales_prediction/
│   ├── cleaning.py              # RawData (load + DQC checks), DataCleaner
│   ├── features.py              # FeatureBuilder, FeaturePipeline
│   ├── split.py                 # ChronologicalSplitter, MonthTimeSeriesSplit, Preprocessor
│   ├── validation.py            # Validator, rmse, ValidationResult
│   ├── models.py                # baseline, Linear Regression, SVR, Random Forest, XGBoost
│   ├── tuning.py                # RandomSearchTuner
│   └── experiment.py            # ModelSelectionExperiment
├── pyproject.toml               # dependencies (managed with uv)
├── uv.lock
├── .pre-commit-config.yaml      # pre-commit hooks (including ruff)
└── README.md
```

## Setup

```powershell
git clone https://github.com/tereshchuk1/sales_prediction.git
cd sales_prediction

# create .venv and install dependencies
uv sync

# enable pre-commit hooks
uv run pre-commit install
```

## Run

```powershell
# raw csv -> data/features.parquet
uv run sales-prediction

# model comparison + hyperparameter tuning + submission (about 1.5 hours)
uv run python -m sales_prediction.experiment --trials 8 --submission
```

## Validation

| Part | Months | Used for |
|---|---|---|
| CV, 3 folds | train: 12 previous months, validation: 30 / 31 / 32 | model comparison, choosing the best model |
| Tuning CV, 2 folds | validation: 31 / 32 | random search of hyperparameters |
| Holdout | train 21-32, validation 33 | checked once at the end, not used for choosing |
| Test | 34 | Kaggle submission (no target) |

- Splits are by month (`MonthTimeSeriesSplit`)
- Sliding window of 12 months: every fold has about the same train size and one full seasonal cycle.
- Every feature of month b uses only months < b; the current month price is dropped.
- `Preprocessor` (medians, scaler, encoders) is fitted inside each fold on the train part only.
- Metric: RMSE with predictions clipped to [0, 20].

## Models

Every model is a class with `fit(train_df)` / `predict(df)` and owns its preprocessing.

| Model | Notes |
|---|---|
| `LastMonthBaseline` | prediction = sales of the pair last month |
| `LinearRegressionModel` | OLS / ridge, scaled features, one-hot categories |
| `SVRModel` | RBF kernel approximated with Nystroem + linear SVR (exact SVR is O(n²) and does not fit 3M rows), 200k sampled rows |
| `RandomForestModel` | 1M sampled rows, each tree sees `max_samples` of them |
| `XGBoostModel` | histogram trees, native NaN handling, all rows |

## Results

RMSE, tuned parameters (`reports/model_comparison.csv`, `reports/best_params.json`):

| Model | CV default | CV tuned (mean ± std) | fold 30 | fold 31 | fold 32 | holdout 33 |
|---|---|---|---|---|---|---|
| **XGBoost** | 0.7705 | **0.7669** ± 0.062 | 0.6984 | 0.7533 | 0.8492 | 0.8850 |
| Random Forest | 0.7739 | 0.7724 ± 0.067 | 0.7003 | 0.7559 | 0.8611 | 0.8809 |
| Linear Regression (ridge α=10) | 0.8120 | 0.8116 ± 0.081 | 0.7222 | 0.7948 | 0.9179 | 0.9414 |
| SVR | 0.9965 | 0.9252 ± 0.084 | 0.8349 | 0.9026 | 1.0380 | 1.0078 |
| Baseline (last month) | 0.9662 | 0.9662 ± 0.073 | 0.8903 | 0.9441 | 1.0640 | 1.1358 |

- XGBoost is the best model by CV and better than Random Forest in every fold; on the holdout the two are within 0.004, which is smaller than the spread between folds.
- The ranking of models is the same in every fold and on the holdout (trees < linear < SVR / baseline), so the validation is stable enough to rank models.
- Error grows towards later months (fold 30 → holdout 33) for all models, including the baseline: the later months are harder, not the models overfitted.
