from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import time
import pandas as pd
import random

from huggingface_hub import login
from datasets import load_dataset, Dataset

import os
login(token=os.environ["HF_TOKEN"])


options = Options()
options.add_argument("--headless")
driver = webdriver.Chrome(options=options)

company = pd.read_excel("SP500CompanyNameTicker.xlsx")
company["Company"] = company["Company"].str.replace("+", "%")

dataset = []

def estrattore(c):
    link = f"https://www.cnbc.com/search/?query={c}&qsearchterm={c}"
    driver.get(link)
    links = []

    # Accetta cookie se presenti
    try:
        botone_accetta = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "onetrust-accept-btn-handler"))
        )
        botone_accetta.click()
    except:
        pass

    # Ordina per data se possibile
    try:
        botone_ordina = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "sortdate"))
        )
        botone_ordina.click()
    except:
        pass

    # Gestione iframe abbonamenti
    try:
        iframe = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "zephrIframeOutcome"))
        )
        driver.switch_to.frame(iframe)
        button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "[data-testid='analytics-click']"))
        )
        button.click()
        driver.switch_to.default_content()
    except:
        pass  

    # Scrolla la pagina per caricare più risultati
    tempo_di_scroll = 5  
    end_time = time.time() + tempo_di_scroll  
    while time.time() < end_time:
        body = driver.find_element(By.TAG_NAME, "body")
        body.send_keys(Keys.PAGE_DOWN)
        time.sleep(random.uniform(1, 2))

    # Trova i link
    links = driver.find_elements(By.CLASS_NAME, "resultlink")

    if links:
        print(f"✅ Trovati {len(links)} link per {c}:")
        for link in links:
            href = link.get_attribute("href")
            dataset.append({
                "Company": c,
                "Link": href
            })
    else:
        print(f"⚠️ Nessun link trovato per {c}")

for c in company["Company"][:5]:
    estrattore(c)
    time.sleep(2)

# Chiudi browser
driver.quit()

df = pd.DataFrame(dataset)
df["Company"] = df["Company"].str.replace("%", " ")

repo_id = "SelmaNajih001/Cnbc_MultiCompany"

# Carica dataset esistente (se c’è)
try:
    old = load_dataset(repo_id, split="train")
    old_df = old.to_pandas()
except:
    old_df = pd.DataFrame()

# Unisci e rimuovi duplicati
all_df = pd.concat([old_df, df]).drop_duplicates(subset=["Company", "Link"])
all_df = all_df.reset_index(drop=True)

# Converte in dataset Hugging Face
final_ds = Dataset.from_pandas(all_df)
final_ds.push_to_hub(repo_id, private=False)

print("🎉 Dataset aggiornato con successo!")


