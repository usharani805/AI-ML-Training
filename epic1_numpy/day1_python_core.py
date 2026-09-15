from functools import reduce
from collections import Counter
import logging
import time
from functools import wraps


def list_comprehension(numbers: list[int]) -> list[int]:
    return [number * 2 for number in numbers]


def dict_comprehension(numbers: list[int]) -> dict[int, int]:
    return {number: number * number for number in numbers}


def set_comprehension(numbers: list[int]) -> set[int]:
    return {number % 2 for number in numbers}


def square_lambda(number: int) -> int:
    square = lambda x: x * x
    return square(number)


def map_numbers(numbers: list[int]) -> list[int]:
    return list(map(lambda x: x * 2, numbers))


def filter_even_numbers(numbers: list[int]) -> list[int]:
    return list(filter(lambda x: x % 2 == 0, numbers))


def reduce_numbers(numbers: list[int]) -> int:
    return reduce(lambda x, y: x + y, numbers, 0)


def sum_numbers(*args: int) -> int:
    return sum(args)


def multiply_numbers(*args: int) -> int:
    result = 1
    for number in args:
        result *= number
    return result


def display_details(**kwargs: str) -> dict[str, str]:
    return kwargs


def fibonacci(n: int):
    a, b = 0, 1

    while a <= n:
        yield a
        a, b = b, a + b


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        logging.info(
            "%s executed in %.6f seconds",
            func.__name__,
            end_time - start_time,
        )

        return result

    return wrapper


def log_call(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(
            "Calling %s with args=%s, kwargs=%s",
            func.__name__,
            args,
            kwargs,
        )

        result = func(*args, **kwargs)
        return result

    return wrapper


def flatten_nested_list(data: list) -> list:
    return [
        item
        for sublist in data
        for item in (
            flatten_nested_list(sublist)
            if isinstance(sublist, list)
            else [sublist]
        )
    ]


def word_frequency(text: str) -> dict[str, int]:
    words = text.lower().split()
    return dict(Counter(words))


@timer
@log_call
def chunk_list(data: list, size: int):
    for i in range(0, len(data), size):
        yield data[i:i + size]


if __name__ == "__main__":
    numbers = [1, 2, 3, 4, 5]

    print("List comprehension:", list_comprehension(numbers))
    print("Dict comprehension:", dict_comprehension(numbers))
    print("Set comprehension:", set_comprehension(numbers))
    print("Lambda:", square_lambda(5))
    print("Map:", map_numbers(numbers))
    print("Filter:", filter_even_numbers(numbers))
    print("Reduce:", reduce_numbers(numbers))

    print("Sum:", sum_numbers(10, 20, 30))
    print("Multiply:", multiply_numbers(2, 3, 4))
    print("Details:", display_details(name="Ravi", department="IT"))

    print("Fibonacci:", list(fibonacci(10)))

    print(
        "Flatten:",
        flatten_nested_list([1, [2, 3], [4, [5, 6]]]),
    )

    print(
        "Word frequency:",
        word_frequency("python is easy python is useful"),
    )

    print(
        "Chunks:",
        list(chunk_list([1, 2, 3, 4, 5], 2)),
    )