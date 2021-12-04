# import flaml
import os
import pickle
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
import sys
import numpy as np

# import pandas as pd
import torch
import torch.nn as nn
from flaml import AutoML, tune
from flaml.model import BaseEstimator
from pandas import DataFrame, Series
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertForSequenceClassification,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForMaskedLM,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
)


class LmSeqsDataset(Dataset):
    """
    用于 sst2 的数据集类.
    Input:
    ------
        max_model_input_size: max seq length
        data_x: `List[np.array[int]]
        data_y: `List[int]
    """

    def __init__(self, data_x, data_y, max_model_input_size=50, min_model_input_size=3):
        self.max_model_input_size = max_model_input_size
        self.min_model_input_size = min_model_input_size

        self.token_ids = np.array(data_x)
        self.labels = np.array(data_y)

        self.check()

        # self.remove_long_sequences()
        # self.remove_empty_sequences()
        self.check()
        self.print_statistics()

    def __getitem__(self, index):
        return (self.token_ids[index], self.labels[index])

    def __len__(self):
        return len(self.labels)

    def check(self):
        """
        Some sanity checks
        """
        assert len(self.token_ids) == len(self.labels)

    def remove_long_sequences(self):
        """
        Sequences that are too long are split by chunk of max_model_input_size.
        """
        max_len = self.max_model_input_size
        # init_size = len(self)
        indices = np.array([len(x) for x in self.token_ids]) <= max_len
        self.token_ids = self.token_ids[indices]
        self.labels = self.labels[indices]
        # new_size = len(self)

    def remove_empty_sequences(self):
        """
        Too short sequences are simply removed. This could be tuned.
        """
        # init_size = len(self)
        indices = np.array([len(x) for x in self.token_ids]) > self.min_model_input_size
        self.token_ids = self.token_ids[indices]
        self.labels = self.labels[indices]
        # new_size = len(self)

    def print_statistics(self):
        """
        Print some statistics on the corpus. Only the master process.
        """
        print(f"{len(self)} sequences")

    def batch_sequences(self, batch):
        """
        补齐所有的token, 使其长度一致.
        """
        # return batch
        token_ids = [torch.tensor(t[0]) for t in batch]
        labels = [t[1] for t in batch]
        # return token_ids, labels
        assert len(token_ids) == len(labels)
        tk_t = torch.nn.utils.rnn.pad_sequence(
            token_ids, batch_first=True, padding_value=0
        )
        tk_l = torch.tensor(labels)
        return tk_t, tk_l


class DistilBertEstimator(BaseEstimator):
    """Dstill Bert Estimator.
    初始化的时候应该输入一个teacher (well-trained), 一个 student.
    然后对于给定的数据X, Y, 找到最佳的超参数使得此时的student能最好地模仿teacher.
    """

    name = "DistilBertEstimator"

    def __init__(self, task="seq-classification", **config):
        super().__init__(task, **config)
        self.trial_id = str(uuid.uuid1().hex)[:8]
        print(f"Initialized {self.trial_id}")

        self._model = None
        self.teacher_class = BertForSequenceClassification
        self.student_class = DistilBertForSequenceClassification

    @classmethod
    def search_space(cls, **params):
        return {
            "learning_rate": {
                "domain": tune.loguniform(lower=1e-6, upper=1e-3),
                "init_value": 1e-5,
            },
            "batch_size": {
                "domain": tune.choice([4, 8, 16, 32]),
                "init_value": 32,
            },
            "gradient_accumulation_steps": {
                "domain": tune.randint(lower=2, upper=60),
                "init_value": 2,
            },
            "weight_decay": {
                "domain": tune.uniform(lower=0.0, upper=0.3),
                "init_value": 0.0,
            },
            "adam_epsilon": {
                "domain": tune.loguniform(lower=1e-8, upper=1e-6),
                "init_value": 1e-6,
            },
            "seed": {
              "domain": tune.chioce(list(range(40,45))),
              "init value": 42
            },
            "global_max_steps":{
                "domain":sys.maxsize,"init_value":sys.maxsize
            },
            "alpha_ce": {
                "domain": tune.uniform(lower=0.0, upper=1.0),
                "init_value": 0.5,
            },
            "alpha_mlm": {
                "domain": tune.uniform(lower=0.0, upper=1.0),
                "init_value": 0.0,
            },  # if mlm, use mlm over clm
            "alpha_clm": {
                "domain": tune.uniform(lower=0.0, upper=1.0),
                "init_value": 0.5,
            },
            "alpha_cos": {
                "domain": tune.uniform(lower=0.0, upper=1.0),
                "init_value": 0.0,
            },
            "alpha_ca": {
                "domain": tune.uniform(lower=0.0, upper=1.0),
                "init_value": 0.1,
            },
        }

    def fit(self, X_train: DataFrame, y_train: Series, budget=None, **kwargs):
        import math
        from torch.optim import AdamW
        from transformers import get_linear_schedule_with_warmup

        n_epoch = 3

        # hyperpremeter start
        temperature = self.params["temperature"]
        learning_rate = self.params["learning_rate"]
        batch_size = self.params["batch_size"]


        gradient_accumulation_steps = self.params["gradient_accumulation_steps"]
        alpha_ce = self.params["alpha_ce"]
        alpha_clm = self.params["alpha_clm"]

        alpha_ca = self.params["alpha_ca"]

        adam_epsilon = self.params["adam_epsilon"]
        weight_decay = self.params["weight_decay"]
        warmup_prop = self.params["warmup_prop"]
        # hyerpremeter end

        teacher_name = "bert-base-uncased"
        teacher = self.teacher_class.from_pretrained(
            teacher_name, output_hidden_states=True
        )

        student_config = "distilbert-base-uncased.json"
        stu_architecture_config = DistilBertConfig.from_pretrained(student_config)
        student = self.student_class(stu_architecture_config)

        student.train()
        teacher.eval()

        assert student.config.vocab_size == teacher.config.vocab_size
        assert student.config.hidden_size == teacher.config.hidden_size
        assert (
            student.config.max_position_embeddings == teacher.config.max_position_embeddings
        )

        student_config = student.config
        # vocab_size = student.config.vocab_size

        dataloader = self._preprocess(X_train, y_train, batch_size=batch_size)

        ce_loss_fct = nn.KLDivLoss(reduction="batchmean")
        lm_loss_fct = nn.CrossEntropyLoss()
        cosine_loss_fct = nn.CosineEmbeddingLoss(reduction="mean")

        num_steps_epoch = len(dataloader)
        num_train_optimization_steps = (
            int(num_steps_epoch / gradient_accumulation_steps * n_epoch) + 1
        )
        warmup_steps = math.ceil(num_train_optimization_steps * warmup_prop)

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in student.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in student.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=learning_rate,
            eps=adam_epsilon,
            betas=(0.9, 0.98),
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_train_optimization_steps,
        )

        n_total_iter = 0
        epoch = 0
        total_loss_epochs = []
        for _ in range(n_epoch):
            total_loss_epoch = 0
            n_iter = 0
            for batch in tqdm(dataloader):
                student_outputs = student(batch[0], output_hidden_states=True)
                teacher_outputs = teacher(batch[0], output_hidden_states=True)

                s_logits, s_h = (
                    student_outputs["logits"],
                    student_outputs["hidden_states"],
                )
                t_logits, t_h = (
                    teacher_outputs["logits"],
                    teacher_outputs["hidden_states"],
                )

                assert s_logits.size() == t_logits.size()

                loss_ce = (
                    ce_loss_fct(
                        nn.functional.log_softmax(s_logits / temperature, dim=-1),
                        nn.functional.softmax(t_logits / temperature, dim=-1),
                    )
                    * (temperature) ** 2
                )
                loss = alpha_ce * loss_ce

                loss_clm = lm_loss_fct(s_logits, batch[1])

                loss += alpha_clm * loss_clm

                dim = s_h[-1].shape[0]
                slh = s_h[-1].view(dim, -1)
                tlh = t_h[-1].view(dim, -1)
                loss_cos = cosine_loss_fct(
                    slh, tlh, target=slh.new(slh.size(0)).fill_(1)
                )
                loss += alpha_ca * loss_cos

                total_loss_epoch += loss.item()

                # Check for NaN
                if (loss != loss).data.any():
                    raise ValueError("NaN detected")
                    # sys.exit(1)

                loss.backward()
                n_iter += 1
                n_total_iter += 1

                if n_iter % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                    break

            total_loss_epochs.append(total_loss_epoch)
            epoch += 1

        self._model = student
        self._model.model_id = self.trial_id
        return total_loss_epochs[-1]

    def _get_best_student(self):
        if self._model:
            print(f"Model id is: {self._model.model_id}")
            return self._model
        else:
            return ValueError("no model")

    def predict_proba(self, X_test):

        y_test_fake = Series(np.zeros(len(X_test)))
        dataloader = self._preprocess(
            X_test, y_test_fake, batch_size=min(512, len(X_test) // 10)
        )
        best_model = self._get_best_student()  #
        probas = []
        for batch in tqdm(dataloader):
            student_outputs = best_model(batch[0])
            proba = nn.functional.softmax(student_outputs["logits"], dim=-1)
            probas.append(proba)

        probas = torch.cat(probas)
        return probas.data.numpy()

    def predict(self, X_test):

        probas = self.predict_proba(X_test)
        return np.argmax(probas, axis=1)

    def _preprocess(self, X_train, y_train, batch_size=10):

        dataset = LmSeqsDataset(
            [np.array(v) for v in X_train["token_ids"].values],
            y_train,
            max_model_input_size=50,
            min_model_input_size=3,
        )

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=dataset.batch_sequences,
        )
        return dataloader


if __name__ == "__main__":


    os.environ["http_proxy"] = "http://127.0.0.1:7890"
    os.environ["https_proxy"] = "http://127.0.0.1:7890"


    X_train, y_train = pickle.load(open("data/dat_train.pkl", "rb"))
    automl = AutoML()
    automl_settings = {
        "time_budget": 60 * 60,  # in seconds
        "metric": "accuracy",
        "task": "seq-classification",
        "log_file_name": "test_distilBert.log",
    }
    e_m = "DistilBertEstimator"
    automl.add_learner(DistilBertEstimator.name, DistilBertEstimator)
    print(automl._state.learner_classes)
    automl.fit(
        X_train=X_train,
        y_train=y_train,
        estimator_list=[DistilBertEstimator.name],
        **automl_settings,
    )
    pass
