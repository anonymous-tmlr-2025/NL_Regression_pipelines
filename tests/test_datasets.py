import pytest


@pytest.mark.parametrize("dataset_name", [
    "test", 
    "jc_penney_products", 
    "online_boat_listings", 
    "california_house_prices"
])
def test_load_dataset_seed_consistency(dataset_name):
    """Test that load_dataset returns consistent results with same seed"""
    from datasets import load_dataset
    
    # Load dataset twice with same seed
    seed = 42
    train1, val1, test1 = load_dataset(dataset_name, seed=seed)
    train2, val2, test2 = load_dataset(dataset_name, seed=seed)
    
    # Check datasets are identical
    assert len(train1) == len(train2)
    assert len(val1) == len(val2)
    assert len(test1) == len(test2)
    assert train1.equals(train2)
    assert val1.equals(val2)
    assert test1.equals(test2)
        
    # Load with different seed should give different result
    train3, val3, test3 = load_dataset(dataset_name, seed=seed+1)
    assert any((train1.iloc[i] != train3.iloc[i]).any() for i in range(len(train1)))
    assert any((val1.iloc[i] != val3.iloc[i]).any() for i in range(len(val1)))

    # But test sets should be the same
    assert test1.equals(test3)
