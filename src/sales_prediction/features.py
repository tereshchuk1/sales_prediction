"""Feature engineering on the (month, shop, item) table

A feature of month b is built only from months < b.
Lags shift whole months forward, history features use strictly earlier months,
the current month price is dropped.
"""

import itertools

import numpy as np
import pandas as pd

from sales_prediction.cleaning import DATA_DIR, TEST_BLOCK, DataCleaner, RawData

BLOCK = "date_block_num"
TARGET = "item_cnt_month"
KEYS = [BLOCK, "shop_id", "item_id"]
START_BLOCK = 12  # first 12 months have no lag 12

# russian public holidays per month (new year 1-8 jan, 23 feb, 8 mar, 1 and 9 may, 12 jun, 4 nov)
HOLIDAYS = {1: 8, 2: 1, 3: 1, 5: 2, 6: 1, 11: 1}


class FeatureBuilder:
    """Builds the monthly (shop, item) grid with leak-free features."""

    def __init__(self, test_block=TEST_BLOCK, start_block=START_BLOCK, clip=(0, 20)):
        self.test_block = test_block
        self.start_block = start_block
        self.clip = clip

    # grid

    def make_grid(self, train, test):
        # every month: all shops x all items of this month, like the test set
        grid = []
        for block in train[BLOCK].unique():
            part = train[train[BLOCK] == block]
            shops = part["shop_id"].unique()
            items = part["item_id"].unique()
            grid.append(np.array(list(itertools.product([block], shops, items))))
        grid = pd.DataFrame(np.vstack(grid), columns=KEYS)

        # monthly sales, the target is clipped into [0, 20] like in the competition
        monthly = train.groupby(KEYS, as_index=False).agg(
            item_cnt_month=("item_cnt_day", "sum"), price=("item_price", "mean")
        )
        grid = grid.merge(monthly, on=KEYS, how="left")
        grid[TARGET] = grid[TARGET].fillna(0).clip(*self.clip)

        # test month, target is unknown (NaN)
        grid = pd.concat([grid, test[KEYS]], ignore_index=True)
        return grid

    # building features functions

    @staticmethod
    def add_lag(df, group, col, lag, name):
        # mean of col per (group, month), taken from month b - lag
        tmp = df.groupby(group + [BLOCK], as_index=False)[col].mean()
        tmp[BLOCK] += lag
        tmp = tmp.rename(columns={col: name})
        df[name] = df[group + [BLOCK]].merge(tmp, how="left")[name].to_numpy("float32")

    @staticmethod
    def take_from_past(df, group, table, col):
        # for every row of df: value of col from the last row of table with
        # the same group and a strictly earlier month (NaN if there is none)
        left = df[group + [BLOCK]].reset_index().sort_values(BLOCK)
        res = pd.merge_asof(
            left,
            table.sort_values(BLOCK),
            on=BLOCK,
            by=group,
            allow_exact_matches=False,
        )
        return res.set_index("index")[col]

    def add_past_mean(self, df, group, col, name):
        # mean of col over all months before b (expanding mean)
        tmp = df.groupby(group + [BLOCK])[col].agg(["sum", "count"])
        tmp = tmp.groupby(level=group).cumsum().reset_index()
        tmp[name] = tmp["sum"] / tmp["count"]
        df[name] = self.take_from_past(
            df, group, tmp[group + [BLOCK, name]], name
        ).astype("float32")

    def add_months_since_sale(self, df, group, name):
        # months since the last month with sales, before b (NaN = never sold before)
        sold = df.loc[df[TARGET] > 0, group + [BLOCK]].drop_duplicates()
        sold["last_sale"] = sold[BLOCK]
        last_sale = self.take_from_past(df, group, sold, "last_sale")
        df[name] = (df[BLOCK] - last_sale).astype("float32")

    @staticmethod
    def add_age(df, col, name):
        # months since the first sale of an item / shop (NaN = cold start);
        # first sales in month b or later give age <= 0 and are masked, so no future info
        first_sale = df[df[TARGET] > 0].groupby(col)[BLOCK].min()
        age = df[BLOCK] - df[col].map(first_sale)
        df[name] = age.where(age > 0).astype("float32")

    def add_calendar(self, df):
        # calendar features as weekends, holidays etc
        rows = []
        for block in range(self.test_block + 1):
            start = pd.Timestamp(2013 + block // 12, block % 12 + 1, 1)
            days = pd.date_range(start, start + pd.offsets.MonthEnd(0))
            rows.append(
                {
                    BLOCK: block,
                    "month": start.month,
                    "quarter": start.quarter,
                    "year": start.year,
                    "days_in_month": len(days),
                    "n_weekend_days": (days.dayofweek >= 5).sum(),
                    "n_holidays": HOLIDAYS.get(start.month, 0),
                    "week_of_year": start.isocalendar()[1],
                }
            )
        cal = pd.DataFrame(rows)
        # cyclical encoding: december and january are neighbours
        cal["month_sin"] = np.sin(2 * np.pi * (cal["month"] - 1) / 12)
        cal["month_cos"] = np.cos(2 * np.pi * (cal["month"] - 1) / 12)
        # peaks from the EDA: new year (december) and a small spring peak (march)
        cal["is_december"] = (cal["month"] == 12).astype(int)
        cal["is_march"] = (cal["month"] == 3).astype(int)
        return df.merge(cal, on=BLOCK, how="left")

    # feature groups

    def add_pair_features(self, df):
        # lags and rolling means of the shop-item pair
        for lag in [1, 2, 3, 4, 5, 6, 12]:
            self.add_lag(df, ["shop_id", "item_id"], TARGET, lag, f"cnt_lag_{lag}")
        df["cnt_roll3_mean"] = df[[f"cnt_lag_{i}" for i in range(1, 4)]].mean(axis=1)
        df["cnt_roll6_mean"] = df[[f"cnt_lag_{i}" for i in range(1, 7)]].mean(axis=1)
        df["cnt_roll3_std"] = df[[f"cnt_lag_{i}" for i in range(1, 4)]].std(axis=1)
        df = df.drop(columns=["cnt_lag_4", "cnt_lag_5"])

        # history of the pair: about half of test pairs were never sold
        self.add_past_mean(df, ["shop_id", "item_id"], TARGET, "pair_cnt_hist_mean")
        self.add_months_since_sale(df, ["shop_id", "item_id"], "pair_months_since_sale")
        return df

    def add_entity_features(self, df):
        # entity aggregations: item sales in other shops, shop sales in general
        for name, group in [
            ("item", ["item_id"]),
            ("shop", ["shop_id"]),
            ("cat", ["item_category_id"]),
        ]:
            for lag in [1, 2, 3]:
                self.add_lag(df, group, TARGET, lag, f"{name}_cnt_lag_{lag}")
            self.add_past_mean(df, group, TARGET, f"{name}_cnt_hist_mean")
            # growth rate between the last two months (trend of shops and categories, EDA 5.3)
            prev = df[f"{name}_cnt_lag_2"]
            df[f"{name}_cnt_growth"] = (
                df[f"{name}_cnt_lag_1"] / prev.where(prev > 0) - 1
            )
        df["item_cnt_roll3_mean"] = df[[f"item_cnt_lag_{i}" for i in range(1, 4)]].mean(
            axis=1
        )

        self.add_lag(
            df, ["shop_id", "item_category_id"], TARGET, 1, "shop_cat_cnt_lag_1"
        )
        self.add_past_mean(
            df, ["shop_id", "item_category_id"], TARGET, "shop_cat_cnt_hist_mean"
        )
        self.add_lag(df, ["city"], TARGET, 1, "city_cnt_lag_1")
        self.add_lag(df, ["channel"], TARGET, 1, "channel_cnt_lag_1")
        return df

    def add_price_features(self, df):
        # price: only past prices, log because price is close to log-normal
        df["log_price"] = np.log1p(df["price"])
        self.add_lag(df, ["item_id"], "log_price", 1, "item_price_lag_1")
        self.add_lag(df, ["item_id"], "log_price", 2, "item_price_lag_2")
        self.add_past_mean(df, ["item_id"], "log_price", "item_price_hist_mean")
        df["item_price_vs_hist"] = df["item_price_lag_1"] - df["item_price_hist_mean"]
        return df

    def add_lifecycle_features(self, df):
        # life cycle: new items and new shops
        self.add_age(df, "item_id", "item_age")
        self.add_age(df, "shop_id", "shop_age")
        df["item_is_new"] = df["item_age"].isna().astype(int)
        return df

    def build(self, train, test, items, shops):
        df = self.make_grid(train, test)
        df = df.merge(items, on="item_id", how="left")
        df = df.merge(shops, on="shop_id", how="left")
        print("grid:", df.shape)

        df = self.add_pair_features(df)
        df = self.add_entity_features(df)
        df = self.add_price_features(df)
        df = self.add_lifecycle_features(df)
        df = self.add_calendar(df)  # 6. calendar

        # never use the current month price
        df = df.drop(columns=["price", "log_price"])
        df = df[df[BLOCK] >= self.start_block]

        float_cols = df.select_dtypes("float64").columns
        df[float_cols] = df[float_cols].astype("float32")
        return df.sort_values(KEYS).reset_index(drop=True)


class FeaturePipeline:
    """raw csv -> checks -> cleaning -> features -> parquet."""

    def __init__(self, data_dir=DATA_DIR, out_dir=DATA_DIR):
        self.raw = RawData(data_dir)
        self.cleaner = DataCleaner()
        self.builder = FeatureBuilder()
        self.out_dir = out_dir

    def run(self):
        raw = self.raw.load().check()
        train = self.cleaner.clean_train(raw.train)
        test = self.cleaner.clean_test(raw.test, train)
        items = self.cleaner.prepare_items(raw.items, raw.categories)
        shops = self.cleaner.prepare_shops(raw.shops)

        df = self.builder.build(train, test, items, shops)
        print("features:", df.shape)
        df.to_parquet(f"{self.out_dir}/features.parquet", index=False)
        # ID is needed to build the submission
        test[["ID", "shop_id", "item_id"]].to_parquet(
            f"{self.out_dir}/test_ids.parquet", index=False
        )
        return df


def main():
    FeaturePipeline().run()


if __name__ == "__main__":
    main()
