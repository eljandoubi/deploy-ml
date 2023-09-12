from starter.ml.model import train_model, inference, compute_model_metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import pytest

@pytest.mark.parametrize("train_model",[train_model])
def test_train(train_model):
    
    try :
        pytest.model = train_model(pytest.X_train, pytest.y_train)
        
    except Exception as err:
        raise err
        
    assert type(pytest.model)==GridSearchCV
    assert type(pytest.model.estimator)==GradientBoostingClassifier
        
        
@pytest.mark.parametrize("inference",[inference])
def test_inference(inference):
    
    try :
        pytest.preds = inference(pytest.model, pytest.X_test)
        
    except Exception as err:
        raise err
    
    assert type(pytest.preds)==np.ndarray
    
    assert len(pytest.preds)==len(pytest.X_test)
    
    assert ((pytest.preds==1)|(pytest.preds==0)).all()
    
    
@pytest.mark.parametrize("compute_model_metrics",[compute_model_metrics])
def test_metrics(compute_model_metrics):
    
    try :
        outs = compute_model_metrics(pytest.y_test, pytest.preds)
     
    except Exception as err:
        raise err
        
    assert len(outs)==3
    
    for o in outs:
        assert type(o)==np.float64
        assert 0<=o<=1