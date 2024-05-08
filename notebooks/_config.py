# Databricks notebook source
# TODO: Update the config variables

catalog = "temp"
schema = "finetuning"
volume_name = "data"
experiment_path = "/Workspace/Users/<your username here>/finetuning"

data_path = f"/Volumes/{catalog}/{schema}/{volume_name}"
pdf_folder = f"{data_path}/raw_data"

# COMMAND ----------


