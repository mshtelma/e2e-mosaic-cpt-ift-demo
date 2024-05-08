# Databricks notebook source
catalog = "temp"
schema = "erni_finetuning"
volume_name = "data"
data_path = f"/Volumes/{catalog}/{schema}/{volume_name}"
pdf_folder = f"{data_path}/raw_data"
secrets_scope = "erni"
experiment_path = "/Workspace/Users/erni.durdevic@databricks.com"

# COMMAND ----------


