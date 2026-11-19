import pandas as pd

DATA_DIR = "data"
TEST_BLOCK = 34  # november 2015

# Decisions from the DQC notebook
SHOP_MERGE = {0: 57, 1: 58, 10: 11}  # same address, different marker / typo
MAX_PRICE = 100000  # removes "Radmin 3 - 522 лиц" (307 980)

# Findings from the EDA notebook
SERVICE_CATEGORIES = ("Доставка товара", "Служебные", "Билеты", "Карты оплаты")


def load_data(data_dir=DATA_DIR):
    train = pd.read_csv(f"{data_dir}/sales_train.csv")
    test = pd.read_csv(f"{data_dir}/test.csv")
    items = pd.read_csv(f"{data_dir}/items.csv")
    shops = pd.read_csv(f"{data_dir}/shops.csv")
    categories = pd.read_csv(f"{data_dir}/item_categories.csv")
    return train, test, items, shops, categories


def check_data(train, test, items, shops, categories):
    # completeness: no missing data
    for df in [train, test, items, shops, categories]:
        assert df.isnull().sum().sum() == 0, "missing values"

    # keys of reference tables are unique (otherwise merges duplicate rows)
    assert not shops["shop_id"].duplicated().any()
    assert not items["item_id"].duplicated().any()
    assert not categories["item_category_id"].duplicated().any()

    # referential integrity
    assert train["shop_id"].isin(shops["shop_id"]).all()
    assert train["item_id"].isin(items["item_id"]).all()
    assert test["shop_id"].isin(shops["shop_id"]).all()
    assert test["item_id"].isin(items["item_id"]).all()
    assert items["item_category_id"].isin(categories["item_category_id"]).all()

    # date_block_num agrees with the date (0 = january 2013)
    date = pd.to_datetime(train["date"], format="%d.%m.%Y")
    block = (date.dt.year - 2013) * 12 + date.dt.month - 1
    assert (block == train["date_block_num"]).all(), "date and date_block_num disagree"

    # timeliness: no missing months
    assert train["date_block_num"].nunique() == train["date_block_num"].max() + 1
    print("all checks passed")


def clean_train(train):
    n = len(train)
    train = train.copy()
    train["shop_id"] = train["shop_id"].replace(SHOP_MERGE)
    train["date"] = pd.to_datetime(train["date"], format="%d.%m.%Y")

    # negative price is invalid, very high price is an outlier
    train = train[(train["item_price"] > 0) & (train["item_price"] < MAX_PRICE)]

    print(f"train: removed {n - len(train)} of {n} rows")
    return train.reset_index(drop=True)


def clean_test(test, train):
    # no rows are removed from test, every ID needs a prediction
    test = test.copy()
    test["shop_id"] = test["shop_id"].replace(SHOP_MERGE)
    test["date_block_num"] = TEST_BLOCK

    unseen = set(test["shop_id"]) - set(train["shop_id"])
    assert not unseen, f"test shops without history: {unseen}"
    return test


def get_city(name):
    name = name.lstrip("!").strip()
    if name.startswith(("Интернет", "Цифровой")):
        return "Online"
    if name.startswith("Выездная"):
        return "Mobile trade"
    if name.startswith("Сергиев Посад"):
        return "Сергиев Посад"
    return name.split(" ")[0]


def prepare_shops(shops):
    shops = shops[~shops["shop_id"].isin(SHOP_MERGE)].copy()
    shops["city"] = shops["shop_name"].apply(get_city)
    # channel: online is the only growing channel
    shops["channel"] = "Offline"
    shops.loc[shops["city"] == "Online", "channel"] = "Online"
    shops.loc[shops["city"] == "Mobile trade", "channel"] = "Mobile trade"
    shops["is_online"] = (shops["channel"] == "Online").astype(int)
    # non-standard formats with high sales per pair
    shops["is_mobile_trade"] = (shops["channel"] == "Mobile trade").astype(int)
    shops["is_sale_point"] = shops["shop_name"].str.contains("Распродажа").astype(int)
    return shops[
        ["shop_id", "city", "channel", "is_online", "is_mobile_trade", "is_sale_point"]
    ]


def prepare_items(items, categories):
    items = items.merge(categories, on="item_category_id")
    # "Игры - PS4" -> "Игры"
    items["category_group"] = items["item_category_name"].str.split(" - ").str[0]
    # service-like cheap goods sold in large amounts
    items["is_service"] = (
        items["item_category_name"].str.startswith(SERVICE_CATEGORIES).astype(int)
    )
    return items[["item_id", "item_category_id", "category_group", "is_service"]]
