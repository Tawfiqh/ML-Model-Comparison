# %%
import pandas as pd


def load_dataframe():
    return pd.read_csv("heart_cancer_data.csv")


# Needs to return a tuple - (X, y)
def load_heart_cancer_dataset(return_df=False):
    df = load_dataframe()
    target_variable = "target"

    if return_df:
        return df

    y = df[target_variable]
    X = df.drop([target_variable], axis=1)
    return X, y


# Uncomment to run this as a cell and interact with the df variable
# df = load_heart_cancer_dataset(return_df=True)
# display(df)
