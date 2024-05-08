# Databricks notebook source
# MAGIC %pip install -r ../requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_config

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.{volume_name}")

# COMMAND ----------

from finreganalytics.dataprep.dataloading import load_and_clean_data, split
from finreganalytics.utils import get_spark


# COMMAND ----------

download_pdfs = [
    # TODO: Add the URLS of pdfs you want to download.
]

import requests
import os

# Create the directory if it doesn't exist
if not os.path.exists(pdf_folder):
    os.makedirs(pdf_folder)

# Download and save the files
for url in download_pdfs:
    file_name = os.path.basename(url)
    file_path = os.path.join(pdf_folder, file_name)
    response = requests.get(url)
    with open(file_path, 'wb') as file:
        file.write(response.content)
        print(f"Downloaded {url} to {file_path}")

# COMMAND ----------

doc_df = load_and_clean_data(pdf_folder)

display(doc_df)

# COMMAND ----------

splitted_df = split(
    doc_df, hf_tokenizer_name="hf-internal-testing/llama-tokenizer", chunk_size=500
)
display(splitted_df)

# COMMAND ----------

splitted_df.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.splitted_documents")

# COMMAND ----------

cpt_df = get_spark().read.table(f"{catalog}.{schema}.splitted_documents")
cpt_train_df, cpt_val_df = cpt_df.select("text").randomSplit([0.95, 0.05])
cpt_train_df.write.mode("overwrite").format("text").save(
    f"{data_path}/training/cpt/text/train"
)
cpt_val_df.write.mode("overwrite").format("text").save(
    f"{data_path}/training/cpt/text/val"
)

# COMMAND ----------

spark.sql(f"select count(1) from text.`{data_path}/training/cpt/text/train`").display()

# COMMAND ----------

spark.sql(f"select count(1) from text.`{data_path}/training/cpt/text/val`").display()

# COMMAND ----------

import os

os.system(f"rm {data_path}/training/cpt/text/val/_committed_*")
os.system(f"rm {data_path}/training/cpt/text/val/_started_*")
os.system(f"rm {data_path}/training/cpt/text/val/_SUCCESS")

# COMMAND ----------

dbutils.fs.ls(f"/Volumes/{catalog}/{schema}/{volume_name}/training/cpt/text/val/")

# COMMAND ----------

os.system(f"rm {data_path}/training/cpt/text/train/_committed_*")
os.system(f"rm {data_path}/training/cpt/text/train/_started_*")
os.system(f"rm {data_path}/training/cpt/text/train/_SUCCESS")

# COMMAND ----------


