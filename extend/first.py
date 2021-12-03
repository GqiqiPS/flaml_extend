from flaml import tune
from flaml.model import BaseEstimator
from pandas import DataFrame, Series
from transformers import DistilBertModel, DistilBertConfig
import uuid

class DistilBertEstimator(BaseEstimator):

    """Dstill Bert Estimator.
    初始化的时候应该输入一个teacher (well-trained), 一个 student.
    然后对于给定的数据X, Y, 找到最佳的超参数使得此时的student能最好地模仿teacher.
    Initialize with teacher well-trained and Student trained.
Then, for the given data X and Y, find the best hyperparameter so that the student can best imitate the teacher.
    """

    def __init__(self, task="distil-bert-opt", **config):
        import uuid
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
        config = {
            "activation": "gelu",
            "attention_dropout": 0.1,
            "dim": 768,
            "dropout": 0.1,
            "hidden_dim": 3072,
            "initializer_range": 0.02,
            "max_position_embeddings": 512,
            "model_type": "distilbert",
            "n_heads": 12,
            "n_layers": 6,
            "pad_token_id": 0,
            "qa_dropout": 0.1,
            "seq_classif_dropout": 0.2,
            "sinusoidal_pos_embds": False,
            "transformers_version": "4.12.5",
            "vocab_size": 30522,
        }
        configuration = DistilBertConfig()
        model = DistilBertModel(configuration)
        pass

    def predict(self, X_test):
        pass

    def cleanup(self):
        del self._model
        self._model = None

