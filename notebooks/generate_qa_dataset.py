# Databricks notebook source
# MAGIC %pip install -r ../requirements.txt
# MAGIC dbutils.library.restartPython()
# COMMAND ----------
import os
from langchain_community.chat_models.databricks import ChatDatabricks

from finreganalytics.dataprep.qagen import build_qa_eval_dataset
from finreganalytics.utils import get_spark
from finreganalytics.dataprep import store_as_mds, store_as_jsonl
from finreganalytics.dataprep.ift_data_prep import prepare_ift_dataset

try:
    context = dbutils.entry_point.getDbutils().notebook().getContext()  # noqa
    os.environ["DATABRICKS_HOST"] = context.apiToken().get()
    os.environ["DATABRICKS_TOKEN"] = context.apiUrl().get()
except:
    pass
# COMMAND ----------


chunks_df = get_spark().read.table("msh.finreg.splitted_documents")
chunks = chunks_df.toPandas()["text"].values.tolist()
# COMMAND ----------

llm_dbrx = ChatDatabricks(endpoint="databricks-dbrx-instruct", temperature=0.1)
EVALUATION_QUESTION_GENERATION_PROMPT_TMPL = """\
Context information is below.

---------------------
{context}
---------------------

Given the context information and not prior knowledge.
generate only questions based on the below query.

You are a Teacher/ Professor in Financial Regulation. Your task is to setup \
{num_questions_per_chunk} questions for an upcoming \
quiz/examination on Capital Requirements Regulation (CRR). The questions should be diverse in nature \
across the document. Restrict the questions to the \
context information provided.
Please generate exactly {num_questions_per_chunk} questions and no more. 
Do not include any further information.

Below is an example of a question.
Always format the output in JSON format as follows:

```json
[ 
"What problems addresses Capital Requirements Regulation?",
"What is Common Reporting Framework (COREP) ?" 
] 
``` """
QA_TEMPLATE_RAG = """
Context information is below.

---------------------
{context}
---------------------

You are an expert in European Financial Regulation. 
You are answering questions related to Financial Regulation for the Financial Institutes in the European Union. 
If the question is not related to one of these topics, kindly decline to answer. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Keep the answer as concise as possible.
Please do not repeat the answer and do not add any additional information. 

Question: {question}

Answer:
"""
# COMMAND ----------
qa_questions_df = build_qa_eval_dataset(
    chunks,
    llm_dbrx,
    question_prompt_template_str=EVALUATION_QUESTION_GENERATION_PROMPT_TMPL,
    answer_prompt_template_str=QA_TEMPLATE_RAG,
    num_questions_per_chunk=10,
)

display(qa_questions_df)  # noqa
# COMMAND ----------
get_spark().createDataFrame(qa_questions_df).write.mode("overwrite").saveAsTable(
    "msh.finreg.qa_dataset"
)
# COMMAND ----------
mds_data_path = "/Volumes/msh/finreg/training/ift/mds/"
jsonl_data_path = "/Volumes/msh/finreg/training/ift/jsonl/"


ift_train_df, ift_val_df = (
    get_spark().table("msh.finreg.qa_dataset").randomSplit([0.99, 0.01])
)
ift_train_df.write.mode("overwrite").saveAsTable("msh.finreg.qa_dataset_train")
ift_val_df.write.mode("overwrite").saveAsTable("msh.finreg.qa_dataset_val")
# COMMAND ----------

ift_completions_train_df = prepare_ift_dataset("msh.finreg.qa_dataset_train", limit=-1)
ift_completions_val_df = prepare_ift_dataset("msh.finreg.qa_dataset_val", limit=-1)

# COMMAND ----------

store_as_mds(ift_completions_train_df, os.path.join(mds_data_path, "train"))
store_as_jsonl(ift_completions_train_df, os.path.join(jsonl_data_path, "train.jsonl"))

store_as_mds(ift_completions_val_df, os.path.join(mds_data_path, "val"))
store_as_jsonl(ift_completions_val_df, os.path.join(jsonl_data_path, "val.jsonl"))
