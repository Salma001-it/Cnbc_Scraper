import time
import random
import pandas as pd
import re
import os
import shutil
import gc
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from newspaper import Config, Article
from datasets import load_dataset, Dataset
from huggingface_hub import login
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset

login(token=os.environ["HF_TOKEN"])

user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
config = Config()
config.browser_user_agent = user_agent
config.request_timeout = 8
options = Options()
options.add_argument("--headless")
driver = webdriver.Chrome(options=options)
company = pd.read_excel("SP500CompanyNameTicker.xlsx")
company["Company"] = company["Company"].str.replace("+", "%")
link_dataset = []

def estrattore(c):
    url = f"https://www.cnbc.com/search/?query={c}&qsearchterm={c}"
    driver.get(url)
    try:
        WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler"))
        ).click()
    except:
        pass

    end_time = time.time() + 5
    
    while time.time() < end_time:
        driver.find_element(By.TAG_NAME, "body").send_keys(Keys.PAGE_DOWN)
        time.sleep(random.uniform(0.5, 1.5))

    links = driver.find_elements(By.CLASS_NAME, "resultlink")
    for l in links:
        href = l.get_attribute("href")
        link_dataset.append({"Company": c, "Link": href})

for c in company["Company"][:5]:
    estrattore(c)
    time.sleep(2)

driver.quit()

df_links = pd.DataFrame(link_dataset)
df_links["Company"] = df_links["Company"].str.replace("%", " ")
repo_id = "SelmaNajih001/Cnbc_MultiCompany"

try:
    old = load_dataset(repo_id, split="train")
    old_df = old.to_pandas()
except:
    old_df = pd.DataFrame()

all_df = pd.concat([old_df, df_links]).drop_duplicates(subset=["Link"])

all_df = all_df.reset_index(drop=True)

to_scrape = all_df[all_df.get("Text", pd.Series()).isna() | (all_df.get("Text", pd.Series()) == "")]
dataset = []

for _, row in to_scrape.iterrows():
    url = row["Link"]
    company_name = row["Company"]
    try:
        article = Article(url, config=config)
        article.download()
        article.parse()
        text = article.text
        title = article.title
        date = article.publish_date
        if text:
            dataset.append({
                "Company": company_name,
                "Link": url,
                "Title": title,
                "Date": date,
                "Text": text
            })
    except Exception as e:
        print(f"Errore con {url}: {e}")

df_articles = pd.DataFrame(dataset)

final_df = pd.concat([all_df, df_articles]).drop_duplicates(subset=["Link","Text"])
final_df = final_df.reset_index(drop=True)

final_ds = Dataset.from_pandas(final_df)
final_ds = final_ds.filter(lambda example: example["Text"] is not None and example["Text"] != "")
final_ds.push_to_hub(repo_id, private=True)
