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
├── data/                    # raw CSV files (git-ignored)
├── src/sales_prediction/    # project package
├── EDA.ipynb                # exploratory data analysis
├── pyproject.toml           # dependencies (managed with uv)
├── uv.lock
├── .pre-commit-config.yaml  # pre-commit hooks (including ruff)
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
