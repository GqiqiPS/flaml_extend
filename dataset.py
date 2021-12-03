from datasets import load_dataset, load_metric
import os

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

task = "cola"
model_checkpoint = "distilbert-base"
batch_size = 16

actual_task = "mnli" if task == "mnli-mm" else task
dataset = load_dataset("glue", actual_task)
metric = load_metric("glue", actual_task)
