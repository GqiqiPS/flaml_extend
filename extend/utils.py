import logging
import argparse
from dataclasses import dataclass, field
from typing import Dict, Any

from ..data import (
    SUMMARIZATION,
    SEQREGRESSION,
    SEQCLASSIFICATION,
    NLG_TASKS
)


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - PID: %(process)d -  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

@dataclass
class DISTILHPOArgs:
    """The HPO setting

    Args:
        output_dir (:obj:`str`):
            data root directory for outputing the log, etc.
        model_path (:obj:`str`, `optional`, defaults to :obj:`facebook/muppet-roberta-base`):
            A string, the path of the language model file, either a path from huggingface
            model card huggingface.co/models, or a local path for the model
        fp16 (:obj:`bool`, `optional`, defaults to :obj:`False`):
            A bool, whether to use FP16
        max_seq_length (:obj:`int`, `optional`, defaults to :obj:`128`):
            An integer, the max length of the sequence
        ckpt_per_epoch (:obj:`int`, `optional`, defaults to :obj:`1`):
            An integer, the number of checkpoints per epoch

    """
    force: bool = field(
        default=False,
        metadata={"action":"store_true", "help":"Overwrite output_dir if it already exists."})

    output_dir: str = field(
        default="data/output/", metadata={"help": "data dir", "required": True}
    )

    student_pretrained_weights_path: str = field(
        default=None,
        metadata={"help": "Load student initialization checkpoint."},)

    temperature: float = field(
        default=2.0,
        metadata={"help": "Temperature for the softmax temperature"}, )

    mlm: bool = field(
        default=True,
        metadata={"action": "store_false", "help": "The LM step: MLM or CLM. If `mlm` is True, the MLM is used over CLM."})

    restrict_ce_to_mask: bool = field(
        default=False,
        metadata={"action": "store_true", "help": "If true, compute the distilation loss only the [MLM] prediction distribution."})

    freeze_pos_embs: bool = field(
        default=False,
        metadata={"action": "store_true", "help": "Freeze positional embeddings during distillation. For student_type in ['roberta', 'gpt2'] only."})

    freeze_token_type_embds: bool = field(
        default=False,
        metadata={"action": "store_true", "help": "Freeze token type embeddings during distillation if existent. For student_type in ['roberta'] only."})

    group_by_size: bool = field(
        default=True,
        metadata={"action": "store_false", "help": "If true, group sequences that have similar length into the same batch. Default is true."})

    gradient_accumulation_steps: int = field(
        default=50,
        metadata={"help": "Gradient accumulation for larger training batches."},
    )

    fp16: bool = field(default=True, metadata={"help": "whether to use the FP16 mode"})

    max_seq_length: int = field(default=128, metadata={"help": "max seq length"})

    ckpt_per_epoch: int = field(default=1, metadata={"help": "checkpoint per epoch"})

    @staticmethod
    def load_args():
        from dataclasses import fields

        arg_parser = argparse.ArgumentParser()
        for each_field in fields(DISTILHPOArgs):
            print(each_field)
            arg_parser.add_argument(
                "--" + each_field.name,
                type=each_field.type,
                help=each_field.metadata["help"],
                required=each_field.metadata["required"]
                if "required" in each_field.metadata
                else False,
                choices=each_field.metadata["choices"]
                if "choices" in each_field.metadata
                else None,
                action=each_field.metadata["action"]
                if "action" in each_field.metadata
                else None,
                default=each_field.default,
            )
        console_args, unknown = arg_parser.parse_known_args()
        return console_args
