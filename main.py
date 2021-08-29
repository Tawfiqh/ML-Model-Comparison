# %%
from models.k_nearest import KNearest
from models.linear_regressor import LinearRegressor
from models.svr import Svr
from models.decision_tree import DecisionTree
from models.random_forest import RandomForest


from models.classification.k_nearest_classification import KNearestClassification
from models.classification.DecisionTreeClassifier import DecisionTreeClassification
from models.classification.GaussianBayes import GaussianNBClassifier
from models.classification.GaussianProcessClassifier import (
    GaussianProcessClassification,
)
from models.classification.LinearDiscriminator import LinearDiscriminantClassifier
from models.classification.LogisticRegression import LogisticRegressionClassifier
from models.classification.RandomForestClassifier import RandomForestClassification
from models.classification.SVC import SVClassifier


from data.get_data import (
    get_boston_train_test_val_datasets,
    get_diabetes_train_test_val_datasets,
    get_school_data_train_test_val_datasets,
    get_car_data_train_test_val_datasets,
    get_breast_cancer_train_test_val_datasets,
    get_heart_cancer_train_test_val_datasets,
    get_iris_train_test_val_datasets,
    get_wine_train_test_val_datasets,
)

import pandas as pd
from time import perf_counter
from datetime import datetime

import matplotlib.pyplot as plt

from os import path as os_path


def run_all_models_on_dataset(
    models, data_set, dataset_name, output_to_csv=False, fit_hyper_parameters=False
):
    all_model_results = []

    for model_name in models.keys():
        print(f"     {model_name}")
        model = models[model_name]
        time_start = perf_counter()

        # Tune (if the model has a function for tuning)

        if fit_hyper_parameters and getattr(model, "find_hyper_paramters", None):
            model.find_hyper_paramters(data_set["train"], data_set["test"])

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

        model_mse_results = model.score_all_mse(
            data_set["train"], data_set["test"], data_set["val"]
        )

        model_mean_absolute_error = model.mae(data_set["whole"])
        # output the result
        if output_to_csv:
            X_df = pd.DataFrame(data_set["whole"][0])
            y = pd.DataFrame(data_set["whole"][1])

            y_hat = model.model.predict(data_set["whole"][0])
            y_hat = pd.DataFrame(y_hat)

            X_df["y"] = y
            X_df["y_hat"] = y_hat
            current_time = datetime.now().strftime("%Y_%b_%d-%H_%M")
            X_df.to_csv(
                os_path.join(
                    "CSV_outputs", f"{model_name}_{dataset_name}_{current_time}.csv"
                )
            )

        # print(f"model_results:{model_results}")
        if model_results:
            all_model_results.append(
                [
                    model_name,
                    fit_time,
                    scoring_time,
                    model_mean_absolute_error,
                    *model_results,
                    *model_mse_results,
                ]
            )

    df = pd.DataFrame(
        all_model_results,
        columns=[
            "model_name",
            "fit_time",
            "scoring_time",
            "mean_absolute_error",
            "training_r^2_score",
            "testing_r^2_score",
            "validation_r^2_score",
            "training_mse_score",
            "testing_mse_score",
            "validation_mse_score",
        ],
    )
    pd.options.display.float_format = "{:,.4f}".format

    if output_to_csv:
        current_time = datetime.now().strftime("%Y_%b_%d-%H_%M")
        df.to_csv(os_path.join("CSV_outputs", f"results_df_{current_time}.csv"))

    print(df)
    print()

    best_result = df[df["validation_r^2_score"] == df["validation_r^2_score"].max()]
    print("Best model result:")
    print(best_result["model_name"].head(1).item())

    # Plot the time taken vs the validation score (scatter)
    # -------------------------------------------
    x = df["fit_time"]
    y = df["validation_r^2_score"]

    plt.ylabel("validation_r^2_score")
    plt.xlabel("fit_time")
    plt.title("Fit Time vs Validation Score")

    plt.scatter(x, y, alpha=0.5)
    plt.show()

    # Plot the time taken to fit AND the validation score (bar)
    # -------------------------------------------
    index = df.index
    bar_width = 0.35
    opacity = 0.8

    rects1 = plt.bar(
        index, df["fit_time"], bar_width, alpha=opacity, color="b", label="Fit Time(s)"
    )

    rects2 = plt.bar(
        index + bar_width,
        df["validation_r^2_score"],
        bar_width,
        alpha=opacity,
        color="g",
        label="validation_r^2_score",
    )

    # plt.xlabel("Dataset")
    plt.ylabel("Time(s) and Score")
    plt.title("Score + Time by model")
    plt.xticks(index + bar_width, df["model_name"])
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Plot the mse scores (bar)
    # -------------------------------------------
    bar_width = 0.25
    index = df.index
    opacity = 0.8
    # "training_mse_score",
    # "testing_mse_score",
    # "validation_mse_score",
    rects1 = plt.bar(
        index,
        df["training_mse_score"],
        bar_width,
        alpha=opacity,
        color="b",
        label="training_mse_score",
    )
    rects2 = plt.bar(
        index + bar_width,
        df["testing_mse_score"],
        bar_width,
        alpha=opacity,
        color="g",
        label="testing_mse_score",
    )

    rects3 = plt.bar(
        index + bar_width * 2,
        df["validation_mse_score"],
        bar_width,
        alpha=opacity,
        color="r",
        label="validation_mse_score",
    )

    plt.ylabel("Score")
    plt.title("Mse scores")
    plt.xticks(index + (bar_width), df["model_name"])
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Plot the mse scores (bar)
    # -------------------------------------------
    bar_width = 0.25
    index = df.index
    opacity = 0.8
    # "training_r^2_score",
    # "testing_r^2_score",
    # "validation_r^2_score",
    rects1 = plt.bar(
        index,
        df["training_r^2_score"],
        bar_width,
        alpha=opacity,
        color="b",
        label="training_r^2_score",
    )
    rects2 = plt.bar(
        index + bar_width,
        df["testing_r^2_score"],
        bar_width,
        alpha=opacity,
        color="g",
        label="testing_r^2_score",
    )

    rects3 = plt.bar(
        index + bar_width * 2,
        df["validation_r^2_score"],
        bar_width,
        alpha=opacity,
        color="r",
        label="validation_r^2_score",
    )

    plt.ylabel("Score")
    plt.title("r^2 scores")
    plt.xticks(index + (bar_width), df["model_name"])
    plt.legend()

    plt.tight_layout()
    plt.show()


def run_all_models_on_classification_dataset(
    models, data_set, dataset_name, output_to_csv=False, fit_hyper_parameters=False
):
    all_model_results = []

    for model_name in models.keys():
        # print()
        # print(f"Model: {model_name}")
        model = models[model_name]
        time_start = perf_counter()

        # Tune (if the model has a function for tuning)

        if fit_hyper_parameters and getattr(model, "find_hyper_paramters", None):
            print("Fitting hyper_parameters")
            model.find_hyper_paramters(data_set["train"], data_set["test"])

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

        model_f1_results = model.score_all_f1(
            data_set["train"], data_set["test"], data_set["val"]
        )

        model_precision = model.score_precision(data_set["val"])
        # output the result
        if output_to_csv:
            X_df = pd.DataFrame(data_set["whole"][0])
            y = pd.DataFrame(data_set["whole"][1])

            y_hat = model.model.predict(data_set["whole"][0])
            y_hat = pd.DataFrame(y_hat)

            X_df["y"] = y
            X_df["y_hat"] = y_hat
            current_time = datetime.now().strftime("%Y_%b_%d-%H_%M")
            X_df.to_csv(
                os_path.join(
                    "CSV_outputs", f"{model_name}_{dataset_name}_{current_time}.csv"
                )
            )

        # print(f"model_results:{model_results}")
        if model_results:
            all_model_results.append(
                [
                    model_name,
                    fit_time,
                    scoring_time,
                    *model_results,
                    *model_f1_results,
                    model_precision,
                ]
            )

    df = pd.DataFrame(
        all_model_results,
        columns=[
            "model_name",
            "fit_time",
            "scoring_time",
            "training score",
            "testing score",
            "validation score",
            "training f1-score",
            "testing f1-score",
            "validation f1-score",
            "model score precision",
        ],
    )
    pd.options.display.float_format = "{:,.4f}".format

    if output_to_csv:
        current_time = datetime.now().strftime("%Y_%b_%d-%H_%M")
        df.to_csv(os_path.join("CSV_outputs", f"results_df_{current_time}.csv"))

    print(df)
    print()

    best_result = df[df["validation f1-score"] == df["validation f1-score"].max()]
    print("Best model result:")
    print(best_result["model_name"].head(1).item())


regression_instead_of_classification = False

if regression_instead_of_classification:
    boston_data_set = get_boston_train_test_val_datasets()
    diabetes_data_set = get_diabetes_train_test_val_datasets()
    school_results_data_set = get_school_data_train_test_val_datasets()
    car_results_data_set = get_car_data_train_test_val_datasets()

    datasets = [
        ("boston_data_set", boston_data_set),
        ("diabetes_data_set", diabetes_data_set),
        ("school_results_data_set", school_results_data_set),
        ("car_results_data_set", car_results_data_set),
    ]

    for data_set_name, data_set in datasets:
        models = {
            "K-Nearest": KNearest(),
            "LinearRegressor": LinearRegressor(),
            "SVR": Svr(),
            "DecisionTree": DecisionTree(),
            "RandomForest": RandomForest(),
        }

        print()
        print(f"EVALUATING {data_set_name}")
        run_all_models_on_dataset(models, data_set, data_set_name, output_to_csv=True)

else:
    breast_cancer_data_set = get_breast_cancer_train_test_val_datasets()
    heart_cancer_data_set = get_heart_cancer_train_test_val_datasets()
    iris_data_set = get_iris_train_test_val_datasets()
    wine_data_set = get_wine_train_test_val_datasets()

    datasets = [
        ("heart_cancer_data_set", heart_cancer_data_set),
        # ("breast_cancer_data_set", breast_cancer_data_set),
        # ("iris_data_set", iris_data_set),
        # ("wine_data_set", wine_data_set),
    ]

    for data_set_name, data_set in datasets:
        models = {
            "K-Nearest": KNearestClassification(),
            "DecisionTree": DecisionTreeClassification(),
            "GaussianNB": GaussianNBClassifier(),
            "GaussianProcess": GaussianProcessClassification(),
            "LinearDiscriminant": LinearDiscriminantClassifier(),
            "LogisticRegression": LogisticRegressionClassifier(),
            "RandomForest": RandomForestClassification(),
            "SVC": SVClassifier(),
        }

        print()
        print(f"EVALUATING {data_set_name}")
        run_all_models_on_classification_dataset(
            models,
            data_set,
            data_set_name,
            output_to_csv=False,
            fit_hyper_parameters=True,
        )


# %%
