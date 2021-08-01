# %%
import pandas as pd


def load_dataframe():
    return pd.read_csv("car_data.csv")


def one_hot_encoding(df, columns):
    df = pd.get_dummies(df, columns=columns, drop_first=True,)
    return df


def frequency_encoding(df_column):
    counts = df_column.value_counts()

    col_length = len(df_column)

    df_column = df_column.apply(lambda classroom: counts[classroom] / col_length)
    # display(df_column)
    return df_column


def one_hot_encoding_from_list_column(df, column_name, enum_list):
    def contains_category(category, category_list):
        if category in category_list.split(","):
            return 1
        return 0

    df[column_name] = df[column_name].fillna("")

    for category in enum_list:
        df[f"{column_name}_{category}"] = df[column_name].apply(
            lambda category_list: contains_category(category, category_list)
        )
    return df


# Needs to return a tuple - (X, y)
def load_cleaned_car_data(return_df=False):
    df = load_dataframe()
    df = one_hot_encoding(
        df,
        [
            "Engine Fuel Type",
            "Transmission Type",
            "Driven_Wheels",
            "Vehicle Size",
            "Vehicle Style",
        ],
    )

    df = one_hot_encoding_from_list_column(
        df,
        "Market Category",
        [
            "High-Performance",
            "Performance",
            "Hybrid",
            "Luxury",
            "Diesel",
            "Factory Tuner",
            "Flex Fuel",
            "Hatchback",
            "Exotic",
            "Crossover",
        ],
    )

    df["Make"] = frequency_encoding(df["Make"])
    df["Model"] = frequency_encoding(df["Model"])

    # Drop extra columns
    df = df.drop(["Market Category"], axis=1)

    # Remove NaNs
    df = df.dropna()

    # Remove duplicates
    df = df.drop_duplicates()
    target_variable = "MSRP"

    if return_df:
        print("outputting to clean CSV")
        df.to_csv("after_cleaning.csv")
        return df

    y = df[target_variable]
    X = df.drop([target_variable], axis=1)
    return X, y


# df = load_cleaned_car_data(return_df=True)
# display(df)


# # Extra data analytics we did when inspecting the data
# # %%
# display(df.isna().sum())  # Check for NaNs
# display(df.describe())
# display(df.dtypes)

# print("DUPLICATES:::")
# duplicated = df.duplicated(keep='last')
# display(df[duplicated])

# No duplicates found?
# Replace float data-types with int


# %%
# import matplotlib.pyplot as plt
# import seaborn as sns

# sns.set(rc={"figure.figsize": (11.7, 8.27)})
# plt.hist(self.df["latest_popularity"], bins=30)
# plt.xlabel("(Current) popularity of all radio tracks played between 2015-2021")
# plt.show()

# # correlation between the features (excluding target)
# # Created a dataframe without the 'popularity' col, since we need to see the correlation between the variables
# track_data = pd.DataFrame(self.data, columns=self.feature_names)

# correlation_matrix = track_data.corr().round(2)
# sns.heatmap(data=correlation_matrix, annot=True)
