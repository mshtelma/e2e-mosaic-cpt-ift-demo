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
import mcli

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
    "train_data_path", f"{data_path}/training/cpt/text/train/", "train_data_path"
)
get_dbutils().widgets.text(
    "val_data_path", f"{data_path}/training/cpt/text/val/", "val_data_path"
)

get_dbutils().widgets.text("training_duration", "1ep", "training_duration")
get_dbutils().widgets.text("learning_rate", "5e-7", "learning_rate")

# COMMAND ----------

base_model = get_dbutils().widgets.get("base_model")
train_data_path = get_dbutils().widgets.get("train_data_path")
val_data_path = get_dbutils().widgets.get("val_data_path")
training_duration = get_dbutils().widgets.get("training_duration")
learning_rate = get_dbutils().widgets.get("learning_rate")

# COMMAND ----------

dbutils.fs.ls(train_data_path)

# COMMAND ----------

from databricks.model_training import foundation_model as fm

run = fm.create(
    model=base_model,
    train_data_path=train_data_path,
    task_type='CONTINUED_PRETRAIN',
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

# mcli.initialize(api_key=get_dbutils().secrets.get(scope=secrets_scope, key="mosaic-token"))


# COMMAND ----------

# from mcli import RunStatus

# run = mcli.create_finetuning_run(
#     model="meta-llama/Llama-2-7b-hf",
#     train_data_path=f"dbfs:{train_data_path}",
#     eval_data_path=f"dbfs:{val_data_path}",
#     save_folder="dbfs:/databricks/mlflow-tracking/{mlflow_experiment_id}/{mlflow_run_id}/artifacts/",
#     task_type="CONTINUED_PRETRAIN",
#     training_duration=training_duration,
#     learning_rate=learning_rate,
#     experiment_tracker={
#         "mlflow": {
#             "experiment_path": f"{experiment_path}/e2e_finreg_domain_adaptation_mosaic",
#             "model_registry_path": f"{catalog}.{schema}.crr_mpt7b8k_cpt_v1",
#         }
#     },
#     disable_credentials_check=True,
# )
# print(f"Started Run {run.name}. The run is in status {run.status}.")

# COMMAND ----------

# mcli.wait_for_run_status(run.name, RunStatus.RUNNING)
# for s in mcli.follow_run_logs(run.name):
#     print(s)
