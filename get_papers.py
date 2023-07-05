import json
from typing import List

import requests
from bs4 import BeautifulSoup


def request(url: str):
    """Request html."""
    r = requests.get(url)
    html = r.text
    return html


def main(year: str, keywords: List[str]):
    """Search papers of a certain year with keywords.

    :param year: (str) Publication year of the papers. E.g., 22.
    :param keywords: (List[str]): List of keywords that the searched papers should contain (case-insensitive).
    """

    items = {}  # {title: item}

    html = request(url=root_url)
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find("main").find_all("table"):
        for tr in tag.tbody.find_all("tr"):
            conference = tr.th.text
            for td in tr.find_all("td"):
                if td.text == year:
                    conference_url = f"{root_url}{td.a['href']}"
                    print(f"Searching papers of {conference} in {year}. URL: {conference_url}")

                    conf_html = request(url=conference_url)
                    conf_soup = BeautifulSoup(conf_html, "html.parser")
                    for tag in conf_soup.find("section", {"id": "main"}).find_all("div", recursive=False):
                        if not tag.has_attr("id"):
                            continue
                        for p in tag.find_all("p"):
                            try:
                                paper_span = p.find_all("span", recursive=False)[1]
                                paper_title = paper_span.strong.a.text
                                paper_authors = list(map(
                                    lambda item: item.text,
                                    paper_span.find_all("a", recursive=False)
                                ))
                                paper_url = f"{root_url}{paper_span.strong.a['href']}"
                            except:
                                print(p.prettify())
                                raise

                            for keyword in keywords:
                                if keyword.lower() in paper_title.lower():
                                    item = {
                                        "conference": conference,
                                        "title": paper_title,
                                        "authors": paper_authors,
                                        "url": paper_url,
                                    }
                                    if paper_title not in items.keys():
                                        items[paper_title] = item

    # Display the searching results.
    print("\nSearched results:\n")
    for item in items.values():
        print(json.dumps(
            item,
            indent=4,
            ensure_ascii=False
        ))


if __name__ == '__main__':
    root_url = "https://aclanthology.org"
    main(
        year="22",  # Year of the papers.
        keywords=["grammatical error correction", "error correction"],  # Keywords.
    )
