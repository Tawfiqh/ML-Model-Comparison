from k_nearest import KNearest
from linear_regressor import LinearRegressor
from get_data import get_boston_train_test_val_datasets
import pandas as pd

models = {"LinearRegressor": LinearRegressor()}


data_set = get_boston_train_test_val_datasets()

all_model_results = []

for model_name in models.keys():
    model = models[model_name]

    # FIT
    model.fit(data_set["train"])

    # SCORE ALL
    model_results = model.score_all(
        data_set["train"], data_set["test"], data_set["val"]
    )
    # print(f"model_results:{model_results}")
    if model_results:
        all_model_results.append([model_name, *model_results])

df = pd.DataFrame(
    all_model_results,
    columns=["model_name", "training_score", "testing_score", "validation_score"],
)
pd.options.display.float_format = "{:,.4f}".format

print(df)
print()

best_result = df[df["validation_score"] == df["validation_score"].max()]
print("Best model result:")
print(best_result["model_name"].head(1).item())