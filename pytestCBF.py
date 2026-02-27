import pytest
import pandas as pd
import numpy as np
from collaborative_filtering import CollaborativeFilteringCF

@pytest.fixture
def mock_data():
    data = {
        'user_id': ["S1", "S1", "S2", "S2", "S3"],
        'item_id': ["C1", "C2", "C2", "C3", "C1"],
        'rating': [7.0, 5.0, 6.0, 4.0, 9.0]  
    }
    df = pd.DataFrame(data)
    return df

@pytest.fixture
def cf_model(mock_data):
    cf = CollaborativeFilteringCF(k_neighbors=2, svd_components=2)
    cf.create_user_item_matrix(mock_data)
    cf.compute_similarities()
    return cf

def test_create_user_item_matrix(mock_data):
    cf = CollaborativeFilteringCF()
    cf.create_user_item_matrix(mock_data)
    assert cf.user_item_df is not None
    assert cf.user_item_matrix is not None
    assert cf.user_item_matrix.shape[0] == 3 # 3 users
    assert cf.user_item_matrix.shape[1] == 3 # 3 items
    
def test_compute_similarities(cf_model):
    assert cf_model.user_sim is not None
    assert cf_model.item_sim is not None
    assert cf_model.user_sim.shape[0] == cf_model.user_item_matrix.shape[0]

def test_predict_user_ratings(cf_model):
    user_pred = cf_model.predict_item_based()
    assert user_pred.shape == cf_model.user_item_matrix.shape
    assert np.all(user_pred >= 0) and np.all(user_pred <= 10)  # Assuming ratings are between 0 and 10
    
def test_predict_item_ratings(cf_model):
    item_pred = cf_model.predict_user_based()
    assert item_pred.shape == cf_model.user_item_matrix.shape
    assert np.all(item_pred >= 0) and np.all(item_pred <= 10)  # Assuming ratings are between 0 and 10

def test_combined_predictions(cf_model):
    user_pred = cf_model.predict_item_based()
    item_pred = cf_model.predict_user_based()
    combined = cf_model.combine_predictions(user_pred, item_pred)
    assert combined.shape == user_pred.shape
    assert np.all(combined >= 0 ) and np.all(combined <= 10)  # Assuming ratings are between 0 and 10

def test_recommend(cf_model):
    user_pred = cf_model.predict_user_based()
    result = cf_model.recommend("S1", user_pred, top_n=2)
    if isinstance(result, pd.DataFrame):
        assert "course_code" in result.columns
        assert len(result) <= 2
    else:
        assert isinstance(result, str)
        
def test_evaluate(cf_model):
    actual = cf_model.user_item_matrix.toarray()
    pred = cf_model.predict_user_based()
    metrics = cf_model.evaluate(actual, pred)
    assert "regression" in metrics
    assert "classification" in metrics
    assert "rmse" in metrics["regression"]
    assert "accuracy" in metrics["classification"]