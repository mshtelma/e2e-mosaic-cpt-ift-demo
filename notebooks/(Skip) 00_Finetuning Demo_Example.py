# Databricks notebook source
# MAGIC %md # Finetuning API Notebook
# MAGIC
# MAGIC This is an example notebook for how to use the finetuning API in the Python SDK (`databricks_genai`). 
# MAGIC
# MAGIC Preliminary docs for PrPr are WIP here:
# MAGIC - Concept page: [Databricks Finetuning APIs](https://pr-14641-aws.dev-docs.dev.databricks.com/large-language-models/fine-tuning-api/index.html)
# MAGIC - How to (SDK): [Create and configure a fine-tuning run using Finetuning APIs](https://pr-14641-aws.dev-docs.dev.databricks.com/large-language-models/fine-tuning-api/create-fine-tune-run.html)
# MAGIC - End to end tutorial (UI) [Create and deploy a fine-tuning run using Finetuning APIs](https://pr-14641-aws.dev-docs.dev.databricks.com/large-language-models/fine-tuning-api/fine-tune-run-tutorial.html)
# MAGIC
# MAGIC Overall, your workflow will look like:
# MAGIC 1. Create finetuning run from the SDK
# MAGIC 2. Go to your MLflow Experiments and monitor the run progress. The default experiment name is the `FinetuningRun.name`. You can also override this for a custom experiment name.
# MAGIC 3. Once the run is finished, you can deploy the model on Model Serving.
# MAGIC 4. Once the model is deployed, you can chat with the model in Playground.
# MAGIC
# MAGIC Please clone this notebook and run the commands.

# COMMAND ----------

# MAGIC %sh pip install databricks-genai --upgrade

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from databricks_genai import finetuning as ft

# COMMAND ----------

# DBTITLE 1,model training with mosaicml/mpt-7b
model = 'databricks/dbrx-base'
train_data_path = '/Volumes/.../.../.../train_data.jsonl'
register_to = 'main.default'
training_duration = '2ep'
learning_rate = '5e-8'
eval_prompts = ['CREATE TABLE ball_is_life ( id number, "pick #" number, "nfl team" text, "player" text, "position" text, "college" text ) -- Using valid SQLite, answer the following questions for the tables provided above. -- who was the only player from kansas state and what was their position?',
                'CREATE TABLE table_3791 ( "Year" text, "Stage" real, "Start of stage" text, "Distance (km)" text, "Category of climb" text, "Stage winner" text, "Nationality" text, "Yellow jersey" text, "Bend" real ) -- Using valid SQLite, answer the following questions for the tables provided above. -- What is every yellow jersey entry for the distance 125?',
                'CREATE TABLE table_43208 ( "8:00" text, "8:30" text, "9:00" text, "9:30" text, "10:00" text ) -- Using valid SQLite, answer the following questions for the tables provided above. -- What aired at 10:00 when Flashpoint aired at 9:30?']

run = ft.create(
  model=model,
  train_data_path=train_data_path,
  register_to=register_to,
  training_duration=training_duration,
  learning_rate=learning_rate,
  eval_prompts=eval_prompts,
)
run

# COMMAND ----------

# Events for current run
ft.get_events(run)

# COMMAND ----------

run.name

# COMMAND ----------

ft.list(limit=3)
