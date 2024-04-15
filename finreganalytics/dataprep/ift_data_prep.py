import json
from typing import List, Dict

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
        -- Context:
        {context}
        ------
        -- Question
        {question}
        ------
        {response_key}
        """
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


def transform_completion_udf(iterator):
    for df in iterator:
        df["prompt"] = df.apply(
            lambda row: format_prompt(row["context"], row["question"]), axis=1
        )
        df["response"] = df.answer
        df = df[["prompt", "response"]]
        yield df


def prepare_ift_dataset(
    table_name: str, limit: int, use_chat_formatting: bool = False
) -> DataFrame:
    sdf = get_spark().read.table(table_name)
    if limit > 0:
        sdf = sdf.limit(limit)
    if use_chat_formatting:
        schema = "messages string"
        func_udf = transform_chat_udf
    else:
        schema = "prompt string, response string"
        func_udf = transform_completion_udf
    transformed_sdf = sdf.mapInPandas(func_udf, schema=schema)
    return transformed_sdf
