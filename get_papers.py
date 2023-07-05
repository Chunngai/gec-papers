#!/usr/bin/env python
# coding: utf-8

# In[25]:


import json

import requests
from bs4 import BeautifulSoup


# In[30]:


root_url = "https://aclanthology.org"
year = "22"
queries=("scoring",)


# In[31]:


def request(url: str):
    r = requests.get(url)
    html = r.text
    return html


# In[32]:


papers = []

html = request(url=root_url)
soup = BeautifulSoup(html, "html.parser")
for tag in soup.find("main").find_all("table"):
    tbody = tag.tbody
    for tr in tbody.find_all("tr"):
        conf_name = tr.th.text
        print(conf_name)
        
        for td in tr.find_all("td"):
            if td.text == year:
                conf_url = f"{root_url}{td.a['href']}"
                print(conf_url)
                
                conf_html = request(url=conf_url)
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
                        
                        for query in queries:
                            if query.lower() in paper_title.lower():
                                papers.append({
                                    "conf": conf_name,
                                    "title": paper_title,
                                    "authors": paper_authors,
                                    "url": paper_url,
                                })


# In[33]:


for item in papers:
    print(json.dumps(item, indent=4, ensure_ascii=False))


# In[ ]:





# In[ ]:




