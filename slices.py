from sklearn.model_selection import train_test_split
from starter.ml.model import inference, compute_model_metrics
from starter.ml.data import process_data
import pandas as pd
import joblib


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


data = pd.read_csv("./data/census.csv", sep=", ", engine='python')


_, test = train_test_split(data, test_size=0.20, random_state=42)

model, encoder, lb = joblib.load("./model/transformers.pkl")

X_test, y_test, * _ = process_data(test, categorical_features=cat_features,
                              label="salary", training=False, encoder=encoder,
                              lb=lb)

preds = inference(model, X_test)

list_res = []

for feature in cat_features:
    for val in test[feature].unique():
        
        mask = test[feature]==val

        precision, recall, fbeta = compute_model_metrics(y_test[mask],
                                                         preds[mask])
        
        list_res.append({"feature":feature,"val":val, "precision":precision,
                   "recall": recall, "fbeta":fbeta})
        
slices = pd.DataFrame(list_res)

slices.to_csv("./slice_output.txt")

print(slices.sample(15))
