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


def next_greater(array: List[Any], target: Any, key=lambda x: x) -> int:
    start = 0
    end = len(array) - 1

    ans = -1
    while start <= end:
        mid = (start + end) // 2

        # Move to right side if target is greater.
        if key(array[mid]) <= key(target):
            start = mid + 1
        # Move left side.
        else:
            ans = mid
            end = mid - 1

    return ans
