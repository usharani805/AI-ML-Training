from day1_python_core import (
    list_comprehension,
    dict_comprehension,
    set_comprehension,
    square_lambda,
    map_numbers,
    filter_even_numbers,
    reduce_numbers,
    sum_numbers,
    multiply_numbers,
    display_details,
    fibonacci,
    flatten_nested_list,
    word_frequency,
    chunk_list,
)


def test_list_comprehension():
    assert list_comprehension([1, 2, 3]) == [2, 4, 6]


def test_dict_comprehension():
    assert dict_comprehension([1, 2, 3]) == {1: 1, 2: 4, 3: 9}


def test_set_comprehension():
    assert set_comprehension([1, 2, 3, 4]) == {0, 1}


def test_square_lambda():
    assert square_lambda(5) == 25


def test_map_numbers():
    assert map_numbers([1, 2, 3]) == [2, 4, 6]


def test_filter_even_numbers():
    assert filter_even_numbers([1, 2, 3, 4]) == [2, 4]


def test_reduce_numbers():
    assert reduce_numbers([1, 2, 3, 4]) == 10


def test_empty_list():
    assert list_comprehension([]) == []


def test_single_element_list():
    assert list_comprehension([5]) == [10]


def test_fibonacci_zero():
    assert list(fibonacci(0)) == [0]


def test_flatten_nested_list():
    assert flatten_nested_list([1, [2, 3], [4, [5]]]) == [1, 2, 3, 4, 5]


def test_word_frequency():
    assert word_frequency("python python java") == {
        "python": 2,
        "java": 1,
    }


def test_chunk_list():
    assert list(chunk_list([1, 2, 3, 4, 5], 2)) == [
        [1, 2],
        [3, 4],
        [5],
    ]


def test_sum_numbers():
    assert sum_numbers(10, 20, 30) == 60


def test_multiply_numbers():
    assert multiply_numbers(2, 3, 4) == 24


def test_display_details():
    assert display_details(name="Ravi", department="IT") == {
        "name": "Ravi",
        "department": "IT",
    }