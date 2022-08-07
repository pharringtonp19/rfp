from rfp.add import total 

def test_total_empty() -> None: 
    assert total([]) == 0.0 

def test_total_single_item() -> None:
    assert total([110.0]) == 110.0

def test_total_with_many_items() -> None:
    assert total([1.0, 2.0, 3.0]) == 6.0 