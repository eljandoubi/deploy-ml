from sklearn.metrics import (fbeta_score, precision_score,
                             recall_score, make_scorer)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    parameters = {
        'n_estimators': [250, 500, 1000],
        'max_depth': [3, 5, 10],
        'min_samples_split': [2, 5, 10],
        'learning_rate': [0.01, 0.05, 0.1],
    }

    f1_metric = make_scorer(fbeta_score, beta=1, zero_division=1)

    clf = GridSearchCV(GradientBoostingClassifier(random_state=42),
                       param_grid=parameters,
                       cv=4,
                       n_jobs=-1,
                       verbose=5,
                       refit=True,
                       scoring=f1_metric
                       )

    clf.fit(X_train, y_train)

    return clf


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    
    return preds
