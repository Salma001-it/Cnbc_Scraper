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

# ===============================
# 0. LOGIN HUGGING FACE
# ===============================
login(token=os.environ["HF_TOKEN"])

# ===============================
# 1. SCRAPING DA CNBC
# ===============================
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

# Pulizia cartelle temporanee di Selenium
shutil.rmtree("/tmp/chrome*", ignore_errors=True)

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
            # Output console minimo
            print(f"Processed article: {title} ({url})")
    except Exception as e:
        print(f"Errore con {url}: {e}")

df_articles = pd.DataFrame(dataset)

final_df = pd.concat([all_df, df_articles]).drop_duplicates(subset=["Link","Text"])
final_df = final_df.reset_index(drop=True)

# Pulizia memoria dei DataFrame intermedi
del df_links, df_articles, all_df, to_scrape, dataset
gc.collect()

final_ds = Dataset.from_pandas(final_df)
final_ds = final_ds.filter(lambda example: example["Text"] is not None and example["Text"] != "")
final_ds.push_to_hub(repo_id, private=True)

# ===============================
# 2. NLP CON MISTRAL
# ===============================
def build_prompt(text):
    return f"""<s>[INST]
You are a market analyst.

Here the news:
{text}

Extract only the most relevant facts that impacted the market. Do NOT summarize or repeat the original content.
Be concise. Just list the points.
If there isn't relevant information, just leave it blank.

Group them under:
Economics / Finance:
- ...

Politics:
- ...
[/INST]</s>""".strip()

def extract_sections(text):
    cleaned = re.sub(r"<s>\[INST\].*?\[/INST\]</s>", "", text, flags=re.DOTALL)
    econ_match = re.search(r"Economics\s*/\s*Finance\s*:\s*(.*?)(?:\n[A-Z][\w\s/]+:|$)", cleaned, flags=re.DOTALL | re.IGNORECASE)
    pol_match = re.search(r"Politics\s*:\s*(.*?)(?:\n[A-Z][\w\s/]+:|$)", cleaned, flags=re.DOTALL | re.IGNORECASE)
    econ_part = econ_match.group(1).strip() if econ_match else ""
    politics_part = pol_match.group(1).strip() if pol_match else ""
    return econ_part, politics_part

# Aggiungi prompt
dataset = final_ds.map(lambda example: {"prompt": build_prompt(example["Text"])})

pipe = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.2",
    device_map="auto"
)
pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id

# Generazione batch con output minimo
results = []
batch_size = 2
for i, out in enumerate(pipe(KeyDataset(dataset, "prompt"), batch_size=batch_size, truncation=True, padding=True, return_full_text=False)):
    generated = out[0]["generated_text"]
    econ, pol = extract_sections(generated)
    results.append({
        "Economics_Finance": econ,
        "Politics": pol
    })
    if (i+1) % 10 == 0:
        print(f"Processed {i+1} batches...")

df_results = pd.DataFrame(results)
final_df_out = pd.concat([dataset.to_pandas(), df_results], axis=1)

# Pulizia memoria
del dataset, results, df_results
gc.collect()

final_ds_out = Dataset.from_pandas(final_df_out)
repo_id2 = "SelmaNajih001/Cnbc_MultiCompany2"
final_ds_out.push_to_hub(repo_id2, private=True)

print("✅ Dataset aggiornato con i campi Economics_Finance e Politics!")
