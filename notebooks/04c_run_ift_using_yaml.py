# Databricks notebook source
# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC

# COMMAND ----------

import os.path
import mcli

from finreganalytics.utils import setup_logging, get_dbutils

setup_logging()

# COMMAND ----------

mcli.initialize(api_key=get_dbutils().secrets.get(scope="msh", key="mosaic-token"))

# COMMAND ----------

from mcli import RunConfig, RunStatus

yaml_config = "../yamls/crr-finreg-ift-mpt7b8k-v1.yaml"
run = mcli.create_run(RunConfig.from_file(yaml_config))
print(f"Started Run {run.name}. The run is in status {run.status}.")

# COMMAND ----------

mcli.wait_for_run_status(run.name, RunStatus.RUNNING)
for s in mcli.follow_run_logs(run.name):
    print(s)
