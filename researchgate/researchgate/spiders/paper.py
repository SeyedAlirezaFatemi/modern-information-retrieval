import scrapy

from ..items import ResearchgateItem


class PaperSpider(scrapy.Spider):
    name = "paper"
    allowed_domains = ["researchgate.net"]
    start_urls = [
        "https://www.researchgate.net/publication/323694313_The_Lottery_Ticket_Hypothesis_Training_Pruned_Neural_Networks/",
        "https://www.researchgate.net/publication/317558625_Attention_Is_All_You_Need",
        "https://www.researchgate.net/publication/328230984_BERT_Pre-training_of_Deep_Bidirectional_Transformers_for_Language_Understanding",
    ]

    def parse(self, response):
        item = ResearchgateItem()
        item["id"] = response.request.url.split("/")[-1].split("_")[0]
        item["title"] = response.css("h1.publication-details__title::text").get()
        item["abstract"] = response.css(
            "div.nova-e-text--size-m.nova-e-text--spacing-auto::text"
        ).get()
        item["date"] = (
            response.css("span.nova-e-text--color-grey-900+::text")
            .get()
            .replace(" · ", "")
        )
        reference_links = response.css(
            "div.nova-v-publication-item__title a.nova-e-link--theme-bare::attr(href)"
        ).getall()
        item["references"] = response.css(
            "div.nova-v-publication-item__title a.nova-e-link--theme-bare::text"
        ).getall()
        item["authors"] = response.css(
            "div.nova-v-person-list-item__title a::text"
        ).getall()
        yield item

    def parse_references(self,):
        pass
