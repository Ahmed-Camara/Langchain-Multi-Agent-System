from src.tools.tools import web_search,scrape_url
from rich import print
res = scrape_url.invoke("https://www.nbcnews.com/artificial-intelligence")
print(res)