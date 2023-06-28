import pandas as pd
import numpy as np

from domain.hmrm_domain import HmrmDomain
from extractor.file_extractor import FileExtractor
from foundation.configuration.input import Input
from loader.file_loader import FileLoader


class HmrmBaseline:
    def __init__(self):
        self.file_extractor = FileExtractor()
        self.hmrm_domain = HmrmDomain()
        self.file_loader = FileLoader()

    def start(self):
        users_checkin_filename = Input.get_instance().inputs["users_checkin_filename"]
        users_checkin = pd.read_csv(users_checkin_filename, index_col=False)

        users_checkin.userid = pd.factorize(users_checkin.userid)[0].astype(int)
        users_checkin.placeid = pd.factorize(users_checkin.placeid)[0].astype(int)
        print(users_checkin["userid"].max(), users_checkin["placeid"].max())

        output_filename = Input.get_instance().inputs["features_filename"]

        self.hmrm_domain.start(users_checkin, 0.5, 7, 50)

        df = pd.DataFrame(
            data=np.concatenate(
                (
                    self.hmrm_domain.context_location_embedding,
                    self.hmrm_domain.target_Location_embedding,
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

        print(sum(df.head(1).isna().sum()))
        df.to_csv(output_filename, index=False)
