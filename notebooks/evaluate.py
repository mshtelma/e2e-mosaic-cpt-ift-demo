# Databricks notebook source
# MAGIC %pip install -r ../requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_config

# COMMAND ----------

from langchain_community.chat_models.databricks import ChatDatabricks
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
import pandas as pd
from finreganalytics.dataprep.evaluation import evaluate_qa_chain
from finreganalytics.utils import get_spark


def build_retrievalqa_zeroshot_chain(prompt_template_str: str, llm: BaseLanguageModel):
    prompt = PromptTemplate(template=prompt_template_str, input_variables=["question"])
    chain = prompt | llm | StrOutputParser()

    return chain


def build_retrievalqa_with_context_chain(
    prompt_template_str: str, llm: BaseLanguageModel
):
    prompt = PromptTemplate(
        template=prompt_template_str, input_variables=["context", "question"]
    )

    chain = prompt | llm | StrOutputParser()

    return chain


# COMMAND ----------

QA_TEMPLATE_ZEROSHOT = """
You are a Regulatory Reporting Assistant. 
Please answer the question as precise as possible.  
If you do not know, just say I don't know.

### Instruction:
Please answer the question:
-- Question
{question}
------

### Response:
"""

QA_TEMPLATE_WITH_CTX = """You are a Regulatory Reporting Assistant. 
Please answer the question as precise as possible using information in context. 
If you do not know, just say I don't know.

### Instruction:
Please answer the question using the given context:
-- Context:
{context}
------
-- Question
{question}
------

### Response:
"""

# COMMAND ----------

llm_mistral = ChatDatabricks(endpoint="databricks-dbrx-instruct", temperature=0.1)
qa_chain_zeroshot = build_retrievalqa_zeroshot_chain(QA_TEMPLATE_ZEROSHOT, llm_mistral)
qa_chain_with_ctx = build_retrievalqa_with_context_chain(
    QA_TEMPLATE_WITH_CTX, llm_mistral
)

# COMMAND ----------

from pyspark.sql.functions import col

val_qa_eval_pdf = pd.read_json(
    path_or_buf=f"{data_path}/training/ift/jsonl/val.jsonl", lines=True
)
val_qa_eval_sdf = (
    get_spark()
    .createDataFrame(val_qa_eval_pdf)
    .alias("v")
    .join(
        get_spark().read.table(f"{catalog}.{schema}.qa_dataset").alias("f"),
        col("f.answer") == col("v.response"),
    )
    .select(col("f.context"), col("f.question"), col("f.answer"))
)
val_qa_eval_df = val_qa_eval_sdf.toPandas()
display(val_qa_eval_df)  # noqa

# COMMAND ----------

eval_results = evaluate_qa_chain(
    val_qa_eval_df,
    ["context", "question"],
    qa_chain_zeroshot,
    "CRR_Mistral_Baseline_ZeroShot",
)
print(f"See evaluation metrics below: \n{eval_results.metrics}")
display(eval_results.tables["eval_results_table"])  # noqa

# COMMAND ----------

eval_results = evaluate_qa_chain(
    val_qa_eval_df,
    ["context", "question"],
    qa_chain_with_ctx,
    "CRR_Mistral_Baseline_With_Ctx",
)
print(f"See evaluation metrics below: \n{eval_results.metrics}")
display(eval_results.tables["eval_results_table"])  # noqa

# COMMAND ----------

llm_mistral = ChatDatabricks(endpoint="erni-fine-tuned", temperature=0.1)
qa_chain_zeroshot = build_retrievalqa_zeroshot_chain(QA_TEMPLATE_ZEROSHOT, llm_mistral)
qa_chain_with_ctx = build_retrievalqa_with_context_chain(
    QA_TEMPLATE_WITH_CTX, llm_mistral
)

# COMMAND ----------

eval_results = evaluate_qa_chain(
    val_qa_eval_df,
    ["context", "question"],
    qa_chain_zeroshot,
    "CRR_Mistral_FT_ZeroShot",
)
print(f"See evaluation metrics below: \n{eval_results.metrics}")
display(eval_results.tables["eval_results_table"])  # noqa

# COMMAND ----------

eval_results = evaluate_qa_chain(
    val_qa_eval_df,
    ["context", "question"],
    qa_chain_with_ctx,
    "CRR_Mistral_FT_With_Ctx",
)
print(f"See evaluation metrics below: \n{eval_results.metrics}")
display(eval_results.tables["eval_results_table"])  # noqa
