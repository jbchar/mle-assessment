import json
import pathlib
import pickle
from typing import List
from typing import Tuple

import pandas
from sklearn import model_selection
from sklearn import neighbors, ensemble
from sklearn import pipeline
from sklearn import preprocessing

SALES_PATH = "data/kc_house_data.csv"  # path to CSV with home sale data
DEMOGRAPHICS_PATH = "data/zipcode_demographics.csv"  # path to CSV with demographics
# List of columns (subset) that will be taken from home sale data
SALES_COLUMN_SELECTION = [
    'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'sqft_above', 'sqft_basement', 'zipcode'
]
OUTPUT_DIR = "model"  # Directory where output artifacts will be saved

def load_data(
    sales_path: str, demographics_path: str, sales_column_selection: List[str]
) -> Tuple[pandas.DataFrame, pandas.Series]:
    """Load the target and feature data by merging sales and demographics.

    Args:
        sales_path: path to CSV file with home sale data
        demographics_path: path to CSV file with home sale data
        sales_column_selection: list of columns from sales data to be used as
            features

    Returns:
        Tuple containg with two elements: a DataFrame and a Series of the same
        length.  The DataFrame contains features for machine learning, the
        series contains the target variable (home sale price).

    """
    data = pandas.read_csv(sales_path,
                           usecols=sales_column_selection,
                           dtype={'zipcode': str})
    # actually use the demographics location constant
    demographics = pandas.read_csv(DEMOGRAPHICS_PATH,
                                   dtype={'zipcode': str})

    merged_data = data.merge(demographics, how="left",
                             on="zipcode").drop(columns="zipcode")
    # Remove the target variable from the dataframe, features will remain
    y = merged_data.pop('price')
    x = merged_data
    return x, y


def main() -> List[dict]:
    """Load data, train model, and export artifacts."""
    x, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)

    # train test split was already being created but not used
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x, y, random_state=42)

    # the existing knn model
    kn_model = pipeline.make_pipeline(preprocessing.RobustScaler(),
                                   neighbors.KNeighborsRegressor()).fit(
                                       x_train, y_train)
    
    #try out a basic GBM regression model
    gb_model = pipeline.make_pipeline(preprocessing.RobustScaler(),
                                    ensemble.GradientBoostingRegressor(
                                        n_estimators=150,
                                        learning_rate=0.1,
                                        max_depth=4,
                                        random_state=42,
                                        loss='squared_error')
                                    ).fit(x_train, y_train)

    # Evaluate the model using the test data
    kn_perf_dict = evaluate_model(kn_model, x_test, y_test)
    gbm_perf_dict = evaluate_model(gb_model, x_test, y_test)
    kn_perf_dict['model'] = 'knn'
    gbm_perf_dict['model'] = 'gbm'
    
    # Print the evaluation metrics
    print('kn_perf', kn_perf_dict)
    print('xg_perf', gbm_perf_dict)

    # pick the model with better mse
    # this would be a separate process in production
    # as we would want to know exactly which model is going to be deployed
    model = kn_model
    if kn_perf_dict['mse'] < gbm_perf_dict['mse']:
        model = gb_model

    output_dir = pathlib.Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    # Output model artifacts: pickled model and JSON list of features
    pickle.dump(model, open(output_dir / "model.pkl", 'wb'))
    json.dump(list(x_train.columns),
              open(output_dir / "model_features.json", 'w'))
    
    json.dump(list([kn_perf_dict, gbm_perf_dict]),
              open(output_dir / "model_perfs.json", 'w'))
    

def evaluate_model(
    model: pipeline.Pipeline, x_test: pandas.DataFrame,
    y_test: pandas.Series) -> dict:
    """Evaluate the model using the test data.
    Args:
        model: trained model
        x_test: test data features
        y_test: test data target variable
    Returns:
        dict: dictionary containing the evaluation metrics
    """

    y_preds = model.predict(x_test)

    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    #Evaluate the model using the test data
    mse = mean_squared_error(y_test, y_preds)
    mae = mean_absolute_error(y_test, y_preds)
    r2 = r2_score(y_test, y_preds)

    return {
        "mse": mse,
        "mae": mae,
        "r2": r2
    }


if __name__ == "__main__":
    main()
