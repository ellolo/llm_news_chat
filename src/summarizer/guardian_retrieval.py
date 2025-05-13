from datetime import datetime
import logging
import requests


class GuardianRetriever:
    """
    A class to retrieve articles from The Guardian API."""

    guardian_url = "https://content.guardianapis.com/search"

    def __init__(self, key: str, url: str = guardian_url):
        if key == "":
            raise ValueError("The Guardian key cannot be empty")
        if url == "":
            raise ValueError("The Guardian URL cannot be empty")
        self._key = key
        self.url = url

    def get_articles(
        self,
        query: str,
        from_date: datetime,
        to_date: datetime = datetime.today(),
        top_k: int = 5,
    ):
        """
        Retrieve articles from The Guardian API.
        Args:
            query: The query to search for articles.
            from_date: The start date for the article search.
            to_date: The end date for the article search.
            top_k: The number of articles to retrieve.
        Returns:
            The list of article URLs from The Guardian.
        """
        if query == "":
            raise ValueError("Query cannot be empty")
        assert from_date <= to_date
        order_by = "relevance"
        params = {
            "q": query,
            "from-date": from_date,
            "to-date": to_date,
            "order-by": order_by,
            "api-key": self._key,
        }
        response = requests.get(self.url, params=params, timeout=10)
        article_urls = [
            result["webUrl"] for result in response.json()["response"]["results"]
        ][:top_k]
        logging.debug("Retrieved urls: %s", article_urls)
        return article_urls
