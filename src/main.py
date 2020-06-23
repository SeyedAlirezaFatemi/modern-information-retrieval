import numpy as np
from bidict import bidict
from elasticsearch import Elasticsearch
from elasticsearch_dsl import (
    Document,
    Keyword,
    Text,
    Integer,
    Float,
)


class Paper(Document):
    title = Text(fields={"raw": Keyword()})
    date = Integer()
    abstract = Text()
    authors = Text()
    references = Text()
    page_rank = Float()

    class Index:
        name = "paper-index"


def crawl(max_papers: int):
    from scrapy.crawler import CrawlerProcess

    from paper_crawler.paper_crawler.spiders.semanticscholar import (
        SemanticscholarSpider,
    )
    process = CrawlerProcess(
        settings={
            "FEEDS": {"crawled_papers.json": {"format": "json"}},
            "LOG_ENABLED": False
            #     "LOG_LEVEL": 'INFO',
        }
    )
    process.crawl(SemanticscholarSpider, max_papers=max_papers)
    process.start()


def connect_to_elastic(elastic_address: str):
    address, port = elastic_address.split(":")
    es = Elasticsearch(hosts=[{"host": address, "port": int(port)}])
    es.indices.create(index="paper-index", ignore=400)
    return es


def load_json():
    import json

    with open("crawled_papers.json") as f:
        items = json.load(f)
    return items


def clear_index(es: Elasticsearch):
    es.indices.delete("paper-index", ignore=404)
    es.indices.create(index="paper-index", ignore=400)


def print_menu():
    print("""Here's what I can do for you:
1. Crawl SemanticScholar
2. Index data in ElasticSearch
3. Compute Page Rank and update ElasticSearch
4. Search in papers
5. HITS
9. Exit
    """)


def index_data():
    items = load_json()
    elastic_address = get_elastic_address()
    es = connect_to_elastic(elastic_address)
    # create the mappings in Elasticsearch
    Paper.init(using=es)
    for item in items:
        paper = Paper(meta={"id": item["id"]}, page_rank=1.0, **item)
        paper.save(using=es)


def get_elastic_address() -> str:
    return input("Enter ElasticSearch address:\n?: ")


def calculate_page_rank():
    alpha = float(input("Enter alpha:\n?: "))
    elastic_address = get_elastic_address()
    es = connect_to_elastic(elastic_address)
    items = load_json()
    all_papers = list(items)
    all_ids = sorted(list(map(lambda x: x["id"], all_papers)))
    p_matrix = np.zeros((len(all_ids), len(all_ids)))
    id_loc = dict()
    for index, paper_id in enumerate(all_ids):
        id_loc[paper_id] = index
    for index, paper_id in enumerate(all_ids):
        paper = Paper.get(id=paper_id, using=es)
        if paper.references is not None:
            for reference_id in paper.references:
                try:
                    p_matrix[index, id_loc[reference_id]] = 1
                except KeyError:
                    continue
    N = len(all_ids)
    v = np.ones((1, N))
    row_sums = np.sum(p_matrix, axis=1, keepdims=True)
    # first part is for rows having nonzero elements
    # second part is for dead-ends
    p_matrix = ((row_sums > 0) * 1) * (
        (1 - alpha) * p_matrix / (row_sums + np.logical_not(row_sums > 0) * 1)
        + alpha * v / N
    ) + (np.logical_not(row_sums > 0) * 1) * v / N
    x0 = np.ones((1, N)) / N
    while True:
        next_state = x0 @ p_matrix
        if np.allclose(next_state, x0, rtol=0.0001):
            break
        x0 = next_state

    for index, paper_id in enumerate(all_ids):
        paper = Paper.get(id=paper_id, using=es)
        paper.update(page_rank=next_state[0, index], using=es)
        paper.save(using=es)


def hits():
    papers = load_json()
    # Give each author an id and store them in this dict
    author_ids = bidict()
    id_counter = 0
    for paper in papers:
        paper_authors = paper["authors"]
        for author in paper_authors:
            if author not in author_ids:
                author_ids[author] = id_counter
                id_counter += 1

    # Map each paper to it's authors' ids
    paper_authors_dict = dict()
    for paper in papers:
        paper_id = paper["id"]
        paper_authors = paper["authors"]
        paper_authors_dict[paper_id] = []
        for author in paper_authors:
            author_id = author_ids[author]
            paper_authors_dict[paper_id].append(author_id)

    # Map each author to the authors he/she has referenced
    author_references = dict()
    for paper in papers:
        paper_authors = paper["authors"]
        paper_references = paper["references"]
        references = []
        for reference_id in paper_references:
            if reference_id in paper_authors_dict:
                references += paper_authors_dict[reference_id]
        for author in paper_authors:
            author_id = author_ids[author]
            if author_id not in author_references:
                author_references[author_id] = []
            author_references[author_id] += references
    num_authors = len(author_ids)
    connectivity_matrix = np.zeros((num_authors, num_authors))
    for author, references in author_references.items():
        for reference in references:
            connectivity_matrix[author, reference] = 1
    a = np.ones(num_authors)
    h = np.ones(num_authors)
    for _ in range(5):
        for i in range(num_authors):
            h[i] = np.sum(connectivity_matrix[i, :] * a)
        for i in range(num_authors):
            a[i] = np.sum(connectivity_matrix[:, i].T * h)
        a /= np.sum(a)
        h /= np.sum(h)
    best_authors = []
    for i in np.argpartition(a, -4)[-4:]:
        best_authors.append((author_ids.inverse[i], a[i]))
    print("Best Authors:")
    best_authors = sorted(best_authors, key=lambda x: -x[1])
    for author, authority in best_authors:
        print(f"{author} : {authority}")


if __name__ == '__main__':
    print("""Welcome! The provided notebook better does whatever it is that this interface must do.
Anyways...    
""")
    while True:
        print_menu()
        choice = int(input("?: "))
        if choice == 9:
            print("Khodahafez!")
            exit(0)
        elif choice == 1:
            max_papers = int(input("Enter max number of crawled papers:\n?: "))
            print("Crawling...")
            crawl(max_papers)
            print("Done!")
        elif choice == 2:
            index_data()
        elif choice == 3:
            calculate_page_rank()
        elif choice == 4:
            print("Please refer to the provided notebook for this part.")
        elif choice == 5:
            hits()
        else:
            print("I didn't get that!")
