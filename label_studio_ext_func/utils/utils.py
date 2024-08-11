import inspect
from datetime import datetime


def find_first(func, iter):
    """
    Finds the first element in an iterable that satisfies a condition.

    Args:
        func (callable): The function that defines the condition. This function should take one argument and return a boolean.
        iter (iterable): The iterable to search.

    Returns:
    The first element in 'iter' that satisfies the condition defined by 'func', or None if no such element is found.
    """
    for i in iter:
        if func(i):
            return i
    return None


def timer(func):
    """
    A decorator that prints the time a function or coroutine takes to execute.
    """
    if inspect.iscoroutinefunction(func):
        async def wraps(*args, **kwargs):
            start = datetime.now()
            result = await func(*args, **kwargs)
            print(datetime.now() - start)
            return result
    else:
        def wraps(*args, **kwargs):
            start = datetime.now()
            result = func(*args, **kwargs)
            print(datetime.now() - start)
            return result

    return wraps


def filter_keys(data, **kwargs):
    """
    Filters the keys in a list of dictionaries.

    Args:
        data (list[dict]): The list of dictionaries to filter.
        **kwargs: Pairs of new_key=old_key. The new_key is the key to use in the new dictionaries, and the old_key is the key to look for in the original dictionaries.

    Returns:
    list[dict]: A list of dictionaries, each containing only the chosen keys. If a key is not found in the original dictionary, its value is set to "-".
    """
    return [
        {new_key: item.get(old_key, "-") for new_key, old_key in kwargs.items()}
        for item in data
    ]
