from typing import Any, List


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
