import functools
import json
from typing import List, Dict

from datasets import load_dataset
from pyspark.sql import SparkSession, DataFrame

from finreganalytics.utils import get_spark

SYSTEM_INSTRUCTION = """You are a Regulatory Reporting Assistant.
Please answer the question as precise as possible using information in context.
If you do not know, just say I don't know. """


def format_prompt(context: str, question: str) -> str:
    INSTRUCTION_KEY = "### Instruction:"
    RESPONSE_KEY = "### Response:"
    PROMPT_FOR_GENERATION_FORMAT = """{intro}
{instruction_key}
{context}
{question}
{response_key}"""
    if context is not None and len(context) > 0:
        context = f"-- Context:\n{context}\n------\n"
    if question is not None and len(question) > 0:
        context = f"-- Question:\n{question}\n------\n"
    return PROMPT_FOR_GENERATION_FORMAT.format(
        intro=SYSTEM_INSTRUCTION,
        instruction_key=INSTRUCTION_KEY,
        context=context,
        question=question,
        response_key=RESPONSE_KEY,
    )


def format_chat_completion(
    context: str, question: str, answer: str
) -> Dict[str, List[Dict[str, str]]]:
    messages = []
    messages.append({"role": "system", "content": SYSTEM_INSTRUCTION})
    messages.append(
        {
            "role": "user",
            "content": f"""Context:\n {context}\n\n Please answer the user question using the given context:\n {question}""",
        }
    )
    messages.append({"role": "assistant", "content": answer})

    return {"messages": messages}


def transform_chat_udf(iterator):
    for df in iterator:
        df["messages"] = df.apply(
            lambda row: json.dumps(
                format_chat_completion(row["context"], row["question"], row["answer"])
            ),
            axis=1,
        )
        df = df[["messages"]]
        yield df


def transform_completion_udf(
    iterator,
    apply_prompt_formatting: bool = True,
    context_col: str = "context",
    question_col: str = "question",
    response_col: str = "answer",
):
    for df in iterator:
        df["prompt"] = df.apply(
            lambda row: (
                format_prompt(row.get(context_col), row.get(question_col))
                if apply_prompt_formatting
                else row[question_col]
            ),
            axis=1,
        )
        df["response"] = df[response_col]
        df = df[["prompt", "response"]]
        yield df


def prepare_ift_dataset(
    table_name: str = None,
    spark_df: DataFrame = None,
    limit: int = -1,
    use_chat_formatting: bool = False,
    apply_prompt_formatting: bool = True,
    context_col: str = "context",
    question_col: str = "question",
    response_col: str = "answer",
) -> DataFrame:
    if table_name is None and spark_df is None:
        raise Exception("Either table_name or spark_df must be provided!")
    if table_name is not None and spark_df is not None:
        raise Exception("Either table_name or spark_df must be provided!")

    if table_name:
        sdf = get_spark().read.table(table_name)
    else:
        sdf = spark_df

    if limit > 0:
        sdf = sdf.limit(limit)
    if use_chat_formatting:
        schema = "messages string"
        func_udf = transform_chat_udf
    else:
        schema = "prompt string, response string"
        func_udf = functools.partial(
            transform_completion_udf,
            apply_prompt_formatting=apply_prompt_formatting,
            context_col=context_col,
            question_col=question_col,
            response_col=response_col,
        )
    transformed_sdf = sdf.mapInPandas(func_udf, schema=schema)
    return transformed_sdf


def load_huggingface_dataset(
    name: str, split: str = "train", limit: int = -1
) -> DataFrame:
    pdf = load_dataset(name, split=split).to_pandas()
    if limit > 0:
        pdf = pdf[:limit]
    sdf = get_spark().createDataFrame(pdf)
    return sdf
