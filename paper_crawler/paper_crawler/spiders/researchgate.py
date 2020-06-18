import scrapy

from ..items import PaperItem


def extract_id_from_url(url: str) -> str:
    return url.split("/")[-1].split("_")[0]


class ResearchgateSpider(scrapy.Spider):
    name = "researchgate_crawler"
    allowed_domains = ["researchgate.net"]
    start_urls = [
        "https://www.researchgate.net/publication/323694313_The_Lottery_Ticket_Hypothesis_Training_Pruned_Neural_Networks",
        "https://www.researchgate.net/publication/317558625_Attention_Is_All_You_Need",
        "https://www.researchgate.net/publication/328230984_BERT_Pre-training_of_Deep_Bidirectional_Transformers_for_Language_Understanding",
    ]
    visited_urls = {
        "323694313_The_Lottery_Ticket_Hypothesis_Training_Pruned_Neural_Networks",
        "317558625_Attention_Is_All_You_Need",
        "328230984_BERT_Pre-training_of_Deep_Bidirectional_Transformers_for_Language_Understanding",
    }
    custom_settings = {
        "USER_AGENT": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36"
    }

    def parse(self, response):
        item = PaperItem()
        item["id"] = extract_id_from_url(response.request.url)
        item["title"] = response.css("h1.publication-details__title::text").get()
        item["abstract"] = response.css(
            "div.nova-e-text--size-m.nova-e-text--spacing-auto::text"
        ).get()
        date = next(
            filter(
                lambda candidate: candidate.startswith(" · "),
                response.css("div.nova-e-text--spacing-xxs span::text").getall(),
            ),
            None,
        )
        if date is not None:
            item["date"] = int(
                next(filter(str.isdigit, date.replace(" · ", "").split()), 0)
            )
        else:
            item["date"] = 0
        item["references"] = list(
            map(
                extract_id_from_url,
                response.css("#references .citation__title a::attr(href)").getall(),
            )
        )
        item["authors"] = response.css(
            "#paper-header .author-list__author-name span::text"
        ).getall()
        yield item

        if len(self.visited_urls) > 20:
            return
        reference_links = response.css(
            "div.js-target-references div.nova-v-publication-item__title a.nova-e-link--theme-bare::attr(href)"
        ).getall()
        num_refs_visited = 0
        for reference_link in reference_links:
            if num_refs_visited > 10:
                return
            if reference_link not in self.visited_urls:
                self.visited_urls.add(reference_link)
                yield response.follow(reference_link, callback=self.parse)
                num_refs_visited += 1
