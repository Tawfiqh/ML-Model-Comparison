# %%
from k_nearest import KNearest
from linear_regressor import LinearRegressor
from svm import Svm_svr

from get_data import (
    get_boston_train_test_val_datasets,
    get_diabetes_train_test_val_datasets,
)
import pandas as pd
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np


models = {
    "K-Nearest": KNearest(),
    "LinearRegressor": LinearRegressor(),
    "SVR": Svm_svr(),
}


def run_all_models_on_dataset(models, data_set):
    all_model_results = []

    for model_name in models.keys():
        model = models[model_name]
        time_start = perf_counter()

        # Tune + FIT
        model.fit(data_set["train"], dataset_train=data_set["test"])

        time_finished_fit = perf_counter()
        fit_time = time_finished_fit - time_start

        # SCORE ALL
        model_results = model.score_all(
            data_set["train"], data_set["test"], data_set["val"]
        )
        time_finished_scoring = perf_counter()
        scoring_time = time_finished_scoring - time_finished_fit

        # print(f"model_results:{model_results}")
        if model_results:
            all_model_results.append(
                [model_name, fit_time, scoring_time, *model_results]
            )

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

    # Plot the time taken vs the validation score
    N = 50
    x = df["fit_time"]
    y = df["validation_score"]

    plt.ylabel("validation_score")
    plt.xlabel("fit_time")
    plt.title("Fit Time vs Validation Score")

    plt.scatter(x, y, alpha=0.5)
    plt.show()

    # create plot
    index = df.index
    bar_width = 0.35
    opacity = 0.8

    rects1 = plt.bar(
        index, df["fit_time"], bar_width, alpha=opacity, color="b", label="Fit Time(s)"
    )

    rects2 = plt.bar(
        index + bar_width,
        df["validation_score"],
        bar_width,
        alpha=opacity,
        color="g",
        label="validation_score",
    )

    # plt.xlabel("Dataset")
    plt.ylabel("Time(s) and Score")
    plt.title("Score + Time by model")
    plt.xticks(index + bar_width, df["model_name"])
    plt.legend()

    # plt.tight_layout()
    # plt.show()


boston_data_set = get_boston_train_test_val_datasets()
diabetes_data_set = get_diabetes_train_test_val_datasets()

datasets = [
    ("boston_data_set", boston_data_set),
    ("diabetes_data_set", diabetes_data_set),
]

for data_set_name, data_set in datasets:
    models = {
        "K-Nearest": KNearest(),
        "LinearRegressor": LinearRegressor(),
        "SVR": Svm_svr(),
    }
    print()
    print(f"EVALUATING {data_set_name}")
    run_all_models_on_dataset(models, data_set)

