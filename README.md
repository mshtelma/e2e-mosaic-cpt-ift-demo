# End-to-end Fine-Tuning of LLMs on Capital Requirements Regulation data

In this demo, we will generate 30k questions and answers related to the Capital Requirements Regulation (CRR), and then do further continued pre-training on the CRR text and related documentation (mostly downloaded from the EBA website), followed by instruction fine-tuning on the generated instructions (30k question-answer pairs generated in the previous step) using Databricks Mosaic Fine-Tuning API and Mosaic Cloud Platform.
