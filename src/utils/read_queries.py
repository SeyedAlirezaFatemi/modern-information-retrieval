import glob
from typing import List, Tuple


def read_queries(query_id: str = "all") -> Tuple[List[str], List[List[int]]]:
    queries = []
    relevants = []
    if query_id == "all":
        for file in sorted(glob.glob("./data/queries/*.txt")):
            with open(file, "r") as query_file:
                queries.append(query_file.read())

        for file in sorted(glob.glob("./data/relevance/*.txt")):
            with open(file, "r") as relevance_file:
                relevants.append(
                    list(int(relevant) for relevant in relevance_file.read().split(","))
                )
    else:
        with open(f"./data/queries/{query_id}.txt") as query_file:
            queries.append(query_file.read())
        with open(f"./data/relevance/{query_id}.txt") as relevance_file:
            relevants.append(
                list(int(relevant) for relevant in relevance_file.read().split(","))
            )

    return queries, relevants
