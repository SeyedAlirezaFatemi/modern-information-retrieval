import glob
from typing import List


def read_queries(query_id: str = "all") -> List[str]:
    queries = []
    if query_id == "all":
        for file in glob.glob("./data/queries/*.txt"):
            with open(file, "r") as query_file:
                queries.append(query_file.read())
    else:
        with open("./data/queries/%s.txt" % (query_id,)) as query_file:
            queries.append(query_file.read())
    return queries
