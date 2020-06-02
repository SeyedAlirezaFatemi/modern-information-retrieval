import scrapy

from ..items import ResearchgateItem


class PaperSpider(scrapy.Spider):
    name = "paper"
    allowed_domains = [
        "researchgate.net"
    ]
    start_urls = [
        "https://www.researchgate.net/publication/323694313_The_Lottery_Ticket_Hypothesis_Training_Pruned_Neural_Networks/",
        "https://www.researchgate.net/publication/317558625_Attention_Is_All_You_Need",
        "https://www.researchgate.net/publication/328230984_BERT_Pre-training_of_Deep_Bidirectional_Transformers_for_Language_Understanding",
    ]

    def parse(self, response):
        print("\n")
        print("HTTP STATUS: " + str(response.status))
        item = ResearchgateItem()
        item["title"] = response.css("title::text").get()
        yield item
