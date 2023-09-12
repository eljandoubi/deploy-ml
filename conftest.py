from sklearn.model_selection import train_test_split
from starter.ml.data import process_data
import pandas as pd
import pytest

def data_plugin():

    data = pd.read_csv("/home/a/deploy-ml/data/census.csv",sep=", ")
    
    
    train, test = train_test_split(data, test_size=0.20, random_state=42)
    
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    
    X_test, y_test, _, _ = process_data(test, categorical_features=cat_features,
                                  label="salary", training=False, encoder=encoder,
                                  lb=lb)
    
    return X_train, y_train, encoder, lb, X_test, y_test, None, None


def pytest_configure():
    
    pytest.X_train, pytest.y_train, pytest.encoder, pytest.lb,\
        pytest.X_test, pytest.y_test, pytest.model,\
            pytest.preads = data_plugin()
        
    
    