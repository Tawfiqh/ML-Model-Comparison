# %%
from scipy.sparse import data
from k_nearest import KNearest
from linear_regressor import LinearRegressor
from svm import Svm_svr
from get_data import get_boston_train_test_val_datasets
import pandas as pd
from time import perf_counter


models = {
    "K-Nearest": KNearest(),
    "LinearRegressor": LinearRegressor(),
    "SVR": Svm_svr(),
}


data_set = get_boston_train_test_val_datasets()

all_model_results = []

for model_name in models.keys():
    model = models[model_name]
    time_start = perf_counter()

    # FIT
    model.fit(data_set["train"])

    time_finished_fit = perf_counter()
    fit_time = time_finished_fit - time_start

    # SCORE ALL
    model_results = model.score_all(
        data_set["train"], data_set["test"], data_set["val"]
    )
    time_finished_scoring = perf_counter()
    scoring_time = time_finished_fit - time_start

    # print(f"model_results:{model_results}")
    if model_results:
        all_model_results.append([model_name, fit_time, scoring_time, *model_results])

df = pd.DataFrame(
    all_model_results,
    columns=[
        "model_name",
        "fit_time",
        "scoring_time",
        "training_score",
        "testing_score",
        "validation_score",
    ],
)
pd.options.display.float_format = "{:,.4f}".format

print(df)
print()

best_result = df[df["validation_score"] == df["validation_score"].max()]
print("Best model result:")
print(best_result["model_name"].head(1).item())
