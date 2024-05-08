# Databricks notebook source
# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

# MAGIC %pip install databricks-genai --upgrade

# COMMAND ----------

dbutils.library.restartPython()


# COMMAND ----------

# MAGIC %run ./_config

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC

# COMMAND ----------

import os.path

from finreganalytics.utils import setup_logging, get_dbutils

setup_logging()

SUPPORTED_INPUT_MODELS = [
    "mosaicml/mpt-30b",
    "mosaicml/mpt-7b-8k",
    "mosaicml/mpt-30b-instruct",
    "mosaicml/mpt-7b-8k-instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mistral-7B-v0.1",
    "mistralai/Mixtral-8x7B-v0.1",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-70b-chat-hf",
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Llama-2-70b-hf",
    "codellama/CodeLlama-7b-hf",
    "codellama/CodeLlama-13b-hf",
    "codellama/CodeLlama-34b-hf",
    "codellama/CodeLlama-7b-Instruct-hf",
    "codellama/CodeLlama-13b-Instruct-hf",
    "codellama/CodeLlama-34b-Instruct-hf",
]
get_dbutils().widgets.combobox(
    "base_model", "meta-llama/Llama-2-7b-hf", SUPPORTED_INPUT_MODELS, "base_model"
)
get_dbutils().widgets.text(
    "train_data_path", f"{data_path}/training/ift/jsonl/train.jsonl", "train_data_path"
)

get_dbutils().widgets.text("training_duration", "10ba", "training_duration")
get_dbutils().widgets.text("learning_rate", "1e-6", "learning_rate")
get_dbutils().widgets.text(
    "custom_weights_path",
    "",
    "custom_weights_path",
)

# COMMAND ----------

base_model = get_dbutils().widgets.get("base_model")
train_data_path = get_dbutils().widgets.get("train_data_path")
training_duration = get_dbutils().widgets.get("training_duration")
learning_rate = get_dbutils().widgets.get("learning_rate")
custom_weights_path = get_dbutils().widgets.get("custom_weights_path")
if len(custom_weights_path) < 1:
    custom_weights_path = None

# COMMAND ----------

custom_weights_path

# COMMAND ----------

from databricks.model_training import foundation_model as fm

run = fm.create(
    model=base_model,
    train_data_path=train_data_path,
    # task_type='INSTRUCTION_FINETUNE',
    task_type='CHAT_COMPLETION',
    register_to=f'{catalog}.{schema}',
    training_duration=training_duration,
    learning_rate=learning_rate,
  )

# COMMAND ----------

fm.get_events(run, follow=True)

# COMMAND ----------

latest_runs = fm.list(limit=2)
latest_runs

# COMMAND ----------


