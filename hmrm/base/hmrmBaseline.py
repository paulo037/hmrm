import pandas as pd
import numpy as np

from .optimization import Optimizer
from ..configuration.data import InputDirectory


class HmrmBaseline:
    def __init__(self, file=None, weight=0.5, K=7, embedding_size=50):
        self.optimizer = Optimizer()
        self.input = InputDirectory(file)
        self.weight = weight
        self.K = K
        self.embedding_size = embedding_size

    def start(self):
        users_checkin_filename = self.input.infile
        users_checkin = pd.read_csv(users_checkin_filename, index_col=False).dropna(
            axis=1
        )

        users_checkin.userid = pd.factorize(users_checkin.userid)[0].astype(int)
        users_checkin.placeid = pd.factorize(users_checkin.placeid)[0].astype(int)

        self.optimizer.start(users_checkin, self.weight, self.K, self.embedding_size)

        df = pd.DataFrame(
            data=np.concatenate(
                (
                    self.optimizer.context_location_embedding,
                    self.optimizer.target_location_embedding,
                ),
                axis=1,
            )
        )
        try:
            values = []
            for i in range(df.shape[0]):
                category = users_checkin[users_checkin["placeid"] == i][
                    "categoryid"
                ].unique()[0]
                values.append(category)

            df["category"] = values
        except:
            pass

        return df
