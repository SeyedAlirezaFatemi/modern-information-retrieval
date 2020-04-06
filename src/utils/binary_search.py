from typing import Any, List


def binary_search(array: List[Any], item: Any, key=lambda x: x) -> int:
    left, right = 0, len(array) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if key(array[mid]) == key(item):
            return mid
        elif key(array[mid]) < key(item):
            left = mid + 1
        else:
            right = mid - 1
    return -1
