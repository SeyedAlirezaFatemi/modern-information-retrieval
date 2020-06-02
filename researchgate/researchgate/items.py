import scrapy


class ResearchgateItem(scrapy.Item):
    id = scrapy.Field()
    title = scrapy.Field()
    authors = scrapy.Field()
    date = scrapy.Field()
    abstract = scrapy.Field()
    references = scrapy.Field()
