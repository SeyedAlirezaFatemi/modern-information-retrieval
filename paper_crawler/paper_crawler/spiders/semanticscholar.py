import scrapy

from ..items import PaperItem


def extract_id_from_url(url: str) -> str:
    return url.split("/")[-1]


class SemanticscholarSpider(scrapy.Spider):
    name = "semanticscholar_crawler"
    allowed_domains = ["semanticscholar.org"]
    start_urls = [
        "https://www.semanticscholar.org/paper/The-Lottery-Ticket-Hypothesis%3A-Training-Pruned-Frankle-Carbin/f90720ed12e045ac84beb94c27271d6fb8ad48cf",
        "https://www.semanticscholar.org/paper/Attention-is-All-you-Need-Vaswani-Shazeer/204e3073870fae3d05bcbc2f6a8e263d9b72e776",
        "https://www.semanticscholar.org/paper/BERT%3A-Pre-training-of-Deep-Bidirectional-for-Devlin-Chang/df2b0e26d0599ce3e70df8a9da02e51594e0e992",
    ]
    visited_urls = {
        "f90720ed12e045ac84beb94c27271d6fb8ad48cf",
        "204e3073870fae3d05bcbc2f6a8e263d9b72e776",
        "df2b0e26d0599ce3e70df8a9da02e51594e0e992",
    }
    custom_settings = {
        "USER_AGENT": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36"
    }

    def parse(self, response):
        item = PaperItem()
        item["id"] = extract_id_from_url(response.request.url)
        item["title"] = response.css("h1::text").get()
        more_abstract = response.css(".key-result__highlight ::text").get()
        item["abstract"] = response.css(".text--preline ::text").get() + (
            more_abstract if more_abstract is not None else ""
        )

        item["date"] = response.css(
            ".paper-meta-item+ .paper-meta-item span span ::text"
        ).get()
        if item["date"] is None:
            item["date"] = ""

        reference_links = response.css(
            "#references .citation__title a::attr(href)"
        ).getall()
        item["references"] = list(map(extract_id_from_url, reference_links))

        item["authors"] = response.css(
            "#paper-header .author-list__author-name span::text"
        ).getall()

        yield item

        if len(self.visited_urls) > 10:
            return
        num_refs_visited = 0
        for reference_link in reference_links:
            if num_refs_visited > 10:
                return
            if reference_link not in self.visited_urls:
                self.visited_urls.add(extract_id_from_url(reference_link))
                yield response.follow(reference_link, callback=self.parse)
                num_refs_visited += 1
