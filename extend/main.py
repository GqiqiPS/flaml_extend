# import flaml
import pickle
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from flaml import tune
from flaml.model import BaseEstimator
from pandas import DataFrame, Series
from transformers import (  # GPT2Config,; GPT2LMHeadModel,; GPT2Tokenizer,; RobertaConfig,; RobertaForMaskedLM,; RobertaTokenizer,
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForMaskedLM,
    DistilBertTokenizer,
)

from distiller import Distiller
from lm_seqs_dataset import LmSeqsDataset

MODEL_CLASSES = {
    "distilbert": (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
    # "roberta": (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
    "bert": (BertConfig, BertForMaskedLM, BertTokenizer),
    # "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
}


@dataclass
class Args:
    data_file: str
    student_config: str
    temperature: float = 2.0
    alpha_ce: float = 0.0
    alpha_mlm: float = 0.0
    alpha_clm: float = 0.5
    alpha_mse: float = 0.0
    alpha_cos: float = 0.0
    mlm_mask_prop: float = 0.15
    word_mask: float = 0.8
    word_keep: float = 0.1
    word_rand: float = 0.1
    mlm_smoothing: float = 0.1
    mlm_smoothing: float = 0.7
    n_epoch: int = 3
    batch_size: int = 5
    gradient_accumulation_steps: int = 50
    warmup_prop: float = 0.05
    weight_decay: float = 0.0
    learning_rate: float = 5e-4
    adam_epsilon: float = 1e-6
    max_grad_norm: float = 5.0
    initializer_range: float = 0.02

    # 无需修改
    seed: int = 56
    log_interval: int = 500
    checkpoint_interval: int = 4000
    teacher_name: str = "The teacher model."
    # 存储路径
    dump_path: str = str(Path("./models").resolve())


class DistilBertEstimator(BaseEstimator):
    """Dstill Bert Estimator.
    初始化的时候应该输入一个teacher (well-trained), 一个 student.
    然后对于给定的数据X, Y, 找到最佳的超参数使得此时的student能最好地模仿teacher.
    """

    def __init__(self, task="distil-bert-opt", **config):
        super().__init__(task, **config)
        self.trial_id = str(uuid.uuid1().hex)[:8]

    # def _join(self, X_train, y_train):
    #     y_train = DataFrame(y_train, columns=["label"], index=X_train.index)
    #     train_df = X_train.join(y_train)
    #     return train_df

    @classmethod
    def search_space(cls, **params):
        return {
            "learning_rate": {
                "domain": tune.loguniform(lower=1e-6, upper=1e-3),
                "init_value": 1e-5,
            },
        }

    def fit(self, X_train: DataFrame, y_train: Series, budget=None, **kwargs):
        args = Args()
        student_config_class, student_model_class, _ = MODEL_CLASSES["distilbert"]
        (
            teacher_config_class,
            teacher_model_class,
            teacher_tokenizer_class,
        ) = MODEL_CLASSES["bert"]
        tokenizer = teacher_tokenizer_class.from_pretrained(args.teacher_name)
        # 去除掉的token id.

        special_tok_ids = {}
        for tok_name, tok_symbol in tokenizer.special_tokens_map.items():
            idx = tokenizer.all_special_tokens.index(tok_symbol)
            special_tok_ids[tok_name] = tokenizer.all_special_ids[idx]

        # TEACHER #
        teacher = teacher_model_class.from_pretrained(
            args.teacher_name, output_hidden_states=True
        )
        stu_architecture_config = student_config_class.from_pretrained(
            args.student_config
        )
        stu_architecture_config.output_hidden_states = True
        # student
        student = student_model_class(stu_architecture_config)

        if args.mlm:
            # logger.info(f"Loading token counts from {args.token_counts} (already pre-computed)")
            with open(args.token_counts, "rb") as fp:
                counts = pickle.load(fp)

            token_probs = np.maximum(counts, 1) ** -args.mlm_smoothing
            for idx in special_tok_ids.values():
                token_probs[idx] = 0.0  # do not predict special tokens
            token_probs = torch.from_numpy(token_probs)
        else:
            token_probs = None
        train_lm_seq_dataset = LmSeqsDataset(params=args, data=data)

        self.model = Distiller(
            params=args,
            dataset=train_lm_seq_dataset,
            token_probs=token_probs,
            student=student,
            teacher=teacher,
        )

    def predict(self, X_test):
        pass

    def cleanup(self):
        del self._model
        self._model = None






# class BertEstimator(BaseEstimator):
#     ITER_HP = "global_max_steps"
#
#     def __init__(self, task="seq-classification", **config):
#         super().__init__(task, **config)
#         import uuid
#
#         self.trial_id = str(uuid.uuid1().hex)[:8]
#
#     def _join(self, X_train, y_train):
#         y_train = DataFrame(y_train, columns=["label"], index=X_train.index)
#         train_df = X_train.join(y_train)
#         return train_df
#
#     @classmethod
#     def search_space(cls, **params):
#         return {
#             "learning_rate": {
#                 "domain": tune.loguniform(lower=1e-6, upper=1e-3),
#                 "init_value": 1e-5,
#             },
#             "num_train_epochs": {
#                 "domain": tune.loguniform(lower=0.1, upper=10.0),
#             },
#             "per_device_train_batch_size": {
#                 "domain": tune.choice([4, 8, 16, 32]),
#                 "init_value": 32,
#             },
#             "warmup_ratio": {
#                 "domain": tune.uniform(lower=0.0, upper=0.3),
#                 "init_value": 0.0,
#             },
#             "weight_decay": {
#                 "domain": tune.uniform(lower=0.0, upper=0.3),
#                 "init_value": 0.0,
#             },
#             "adam_epsilon": {
#                 "domain": tune.loguniform(lower=1e-8, upper=1e-6),
#                 "init_value": 1e-6,
#             },
#             "seed": {"domain": tune.choice(list(range(40, 45))), "init_value": 42},
#             "global_max_steps": {"domain": sys.maxsize, "init_value": sys.maxsize},
#         }
#
#     def _init_hpo_args(self, automl_fit_kwargs: dict = None):
#         from .nlp.utils import HPOArgs
#
#         custom_hpo_args = HPOArgs()
#         for key, val in automl_fit_kwargs["custom_hpo_args"].items():
#             assert (
#                     key in custom_hpo_args.__dict__
#             ), "The specified key {} is not in the argument list of flaml.nlp.utils::HPOArgs".format(
#                 key
#             )
#             setattr(custom_hpo_args, key, val)
#         self.custom_hpo_args = custom_hpo_args
#
#     def _preprocess(self, X, task, **kwargs):
#         from .nlp.utils import tokenize_text
#
#         if X.dtypes[0] == "string":
#             return tokenize_text(X, task, self.custom_hpo_args)
#         else:
#             return X
#
#     def fit(self, X_train: DataFrame, y_train: Series, budget=None, **kwargs):
#         from transformers import EarlyStoppingCallback
#         from transformers.trainer_utils import set_seed
#         from transformers import AutoTokenizer, TrainingArguments
#         import transformers
#         from datasets import Dataset
#         from .nlp.utils import (
#             get_num_labels,
#             separate_config,
#             load_model,
#             compute_checkpoint_freq,
#             get_trial_fold_name,
#             date_str,
#         )
#         from .nlp.huggingface.trainer import TrainerForAuto
#
#         this_params = self.params
#
#         class EarlyStoppingCallbackForAuto(EarlyStoppingCallback):
#             def on_train_begin(self, args, state, control, **callback_kwargs):
#                 self.train_begin_time = time.time()
#
#             def on_step_begin(self, args, state, control, **callback_kwargs):
#                 self.step_begin_time = time.time()
#
#             def on_step_end(self, args, state, control, **callback_kwargs):
#                 if state.global_step == 1:
#                     self.time_per_iter = time.time() - self.step_begin_time
#                 if (
#                         budget
#                         and (
#                         time.time() + self.time_per_iter
#                         > self.train_begin_time + budget
#                 )
#                         or state.global_step >= this_params[TransformersEstimator.ITER_HP]
#                 ):
#                     control.should_training_stop = True
#                     control.should_save = True
#                     control.should_evaluate = True
#                 return control
#
#             def on_epoch_end(self, args, state, control, **callback_kwargs):
#                 if (
#                         control.should_training_stop
#                         or state.epoch + 1 >= args.num_train_epochs
#                 ):
#                     control.should_save = True
#                     control.should_evaluate = True
#
#         set_seed(self.params.get("seed", TrainingArguments.seed))
#
#         self._init_hpo_args(kwargs)
#         self._metric_name = kwargs["metric"]
#         if hasattr(self, "use_ray") is False:
#             self.use_ray = kwargs["use_ray"]
#
#         X_val = kwargs.get("X_val")
#         y_val = kwargs.get("y_val")
#
#         X_train = self._preprocess(X_train, self._task, **kwargs)
#         train_dataset = Dataset.from_pandas(self._join(X_train, y_train))
#         if X_val is not None:
#             X_val = self._preprocess(X_val, self._task, **kwargs)
#             eval_dataset = Dataset.from_pandas(self._join(X_val, y_val))
#         else:
#             eval_dataset = None
#
#         tokenizer = AutoTokenizer.from_pretrained(
#             self.custom_hpo_args.model_path, use_fast=True
#         )
#
#         num_labels = get_num_labels(self._task, y_train)
#
#         training_args_config, per_model_config = separate_config(self.params)
#         this_model = load_model(
#             checkpoint_path=self.custom_hpo_args.model_path,
#             task=self._task,
#             num_labels=num_labels,
#             per_model_config=per_model_config,
#         )
#         ckpt_freq = compute_checkpoint_freq(
#             train_data_size=len(X_train),
#             custom_hpo_args=self.custom_hpo_args,
#             num_train_epochs=training_args_config.get(
#                 "num_train_epochs", TrainingArguments.num_train_epochs
#             ),
#             batch_size=training_args_config.get(
#                 "per_device_train_batch_size",
#                 TrainingArguments.per_device_train_batch_size,
#             ),
#         )
#
#         local_dir = os.path.join(
#             self.custom_hpo_args.output_dir, "train_{}".format(date_str())
#         )
#
#         if not self.use_ray:
#             # if self.params = {}, don't include configuration in trial fold name
#             trial_dir = get_trial_fold_name(local_dir, self.params, self.trial_id)
#         else:
#             import ray
#
#             trial_dir = ray.tune.get_trial_dir()
#
#         if transformers.__version__.startswith("3"):
#             training_args = TrainingArguments(
#                 report_to=[],
#                 output_dir=trial_dir,
#                 do_train=True,
#                 do_eval=True,
#                 eval_steps=ckpt_freq,
#                 evaluate_during_training=True,
#                 save_steps=ckpt_freq,
#                 save_total_limit=0,
#                 fp16=self.custom_hpo_args.fp16,
#                 load_best_model_at_end=True,
#                 **training_args_config,
#             )
#         else:
#             from transformers import IntervalStrategy
#
#             training_args = TrainingArguments(
#                 report_to=[],
#                 output_dir=trial_dir,
#                 do_train=True,
#                 do_eval=True,
#                 per_device_eval_batch_size=1,
#                 eval_steps=ckpt_freq,
#                 evaluation_strategy=IntervalStrategy.STEPS,
#                 save_steps=ckpt_freq,
#                 save_total_limit=0,
#                 fp16=self.custom_hpo_args.fp16,
#                 load_best_model_at_end=True,
#                 **training_args_config,
#             )
#
#         def _model_init():
#             return load_model(
#                 checkpoint_path=self.custom_hpo_args.model_path,
#                 task=self._task,
#                 num_labels=num_labels,
#                 per_model_config=per_model_config,
#             )
#
#         self._model = TrainerForAuto(
#             model=this_model,
#             args=training_args,
#             model_init=_model_init,
#             train_dataset=train_dataset,
#             eval_dataset=eval_dataset,
#             tokenizer=tokenizer,
#             compute_metrics=self._compute_metrics_by_dataset_name,
#             callbacks=[EarlyStoppingCallbackForAuto],
#         )
#
#         setattr(self._model, "_use_ray", self.use_ray)
#         self._model.train()
#
#         self.params[self.ITER_HP] = self._model.state.global_step
#         self._checkpoint_path = self._select_checkpoint(self._model)
#
#         self._kwargs = kwargs
#         self._num_labels = num_labels
#         self._per_model_config = per_model_config
#
#         self._ckpt_remains = list(self._model.ckpt_to_metric.keys())
#
#     def _delete_one_ckpt(self, ckpt_location):
#         if self.use_ray is False:
#             try:
#                 shutil.rmtree(ckpt_location)
#             except FileNotFoundError:
#                 logger.warning("checkpoint {} not found".format(ckpt_location))
#
#     def cleanup(self):
#         if hasattr(self, "_ckpt_remains"):
#             for each_ckpt in self._ckpt_remains:
#                 self._delete_one_ckpt(each_ckpt)
#
#     def _select_checkpoint(self, trainer):
#         from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
#
#         if trainer.ckpt_to_metric:
#             best_ckpt, _ = min(
#                 trainer.ckpt_to_metric.items(), key=lambda x: x[1]["val_loss"]
#             )
#             best_ckpt_global_step = trainer.ckpt_to_global_step[best_ckpt]
#             for each_ckpt in list(trainer.ckpt_to_metric):
#                 if each_ckpt != best_ckpt:
#                     del trainer.ckpt_to_metric[each_ckpt]
#                     del trainer.ckpt_to_global_step[each_ckpt]
#                     self._delete_one_ckpt(each_ckpt)
#         else:
#             best_ckpt_global_step = trainer.state.global_step
#             best_ckpt = os.path.join(
#                 trainer.args.output_dir,
#                 f"{PREFIX_CHECKPOINT_DIR}-{best_ckpt_global_step}",
#             )
#         self.params[self.ITER_HP] = best_ckpt_global_step
#         print(trainer.state.global_step)
#         print(trainer.ckpt_to_global_step)
#         return best_ckpt
#
#     def _compute_metrics_by_dataset_name(self, eval_pred):
#         from .ml import sklearn_metric_loss_score
#         from .data import SEQREGRESSION
#         import datasets
#         from .nlp.utils import load_default_huggingface_metric_for_task
#
#         predictions, labels = eval_pred
#         predictions = (
#             np.squeeze(predictions)
#             if self._task == SEQREGRESSION
#             else np.argmax(predictions, axis=1)
#         )
#
#         if isinstance(self._metric_name, str):
#             return {
#                 "val_loss": sklearn_metric_loss_score(
#                     metric_name=self._metric_name, y_predict=predictions, y_true=labels
#                 )
#             }
#         else:
#             (
#                 default_metric_name,
#                 default_metric_mode,
#             ) = load_default_huggingface_metric_for_task(self._task)
#             metric = datasets.load_metric(default_metric_name)
#             multiplier = -1 if default_metric_mode == "max" else 1
#             return {
#                 "val_loss": metric.compute(predictions=predictions, references=labels)[
#                                 default_metric_name
#                             ]
#                             * multiplier
#             }
#
#     def predict_proba(self, X_test):
#         from datasets import Dataset
#         from .nlp.huggingface.trainer import TrainerForAuto
#         from transformers import TrainingArguments
#         from .nlp.utils import load_model
#
#         assert (
#                 self._task in CLASSIFICATION
#         ), "predict_proba is only available in classification tasks"
#
#         X_test = self._preprocess(X_test, self._task, **self._kwargs)
#         test_dataset = Dataset.from_pandas(X_test)
#
#         best_model = load_model(
#             checkpoint_path=self._checkpoint_path,
#             task=self._task,
#             num_labels=self._num_labels,
#             per_model_config=self._per_model_config,
#         )
#         training_args = TrainingArguments(
#             per_device_eval_batch_size=1,
#             output_dir=self.custom_hpo_args.output_dir,
#         )
#         self._model = TrainerForAuto(model=best_model, args=training_args)
#         predictions = self._model.predict(test_dataset)
#         return predictions.predictions
#
#     def predict(self, X_test):
#         from datasets import Dataset
#         from transformers import TrainingArguments
#         from .nlp.utils import load_model
#         from .nlp.huggingface.trainer import TrainerForAuto
#
#         X_test = self._preprocess(X_test, self._task, **self._kwargs)
#         test_dataset = Dataset.from_pandas(X_test)
#
#         best_model = load_model(
#             checkpoint_path=self._checkpoint_path,
#             task=self._task,
#             num_labels=self._num_labels,
#             per_model_config=self._per_model_config,
#         )
#         training_args = TrainingArguments(
#             per_device_eval_batch_size=1,
#             output_dir=self.custom_hpo_args.output_dir,
#         )
#         self._model = TrainerForAuto(model=best_model, args=training_args)
#         predictions = self._model.predict(test_dataset)
#
#         return np.argmax(predictions.predictions, axis=1)
#
#     def config2params(cls, config: dict) -> dict:
#         params = config.copy()
#         params[TransformersEstimator.ITER_HP] = params.get(
#             TransformersEstimator.ITER_HP, sys.maxsize
#         )
#         return params