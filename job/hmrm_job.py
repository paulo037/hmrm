import pandas as pd
import numpy as np

from domain.hmrm_domain import HmrmDomain
from extractor.file_extractor import FileExtractor
from foundation.configuration.input import Input 
from loader.file_loader import FileLoader

class HmrmBaseline():

    def __init__(self):
        self.file_extractor = FileExtractor()
        self.hmrm_domain = HmrmDomain()
        self.file_loader = FileLoader()

    def start(self):
        users_checkin_filename = Input.get_instance().inputs['users_checkin_filename']
        users_checkin = self.file_extractor.read_csv(users_checkin_filename)
        weeplaces = Input.get_instance().inputs["weeplaces?"]
        output_filename = Input.get_instance().inputs["features_filename"]
        #----------------------
        
        """
        start t
        """        

        self.hmrm_domain.start(users_checkin, 0.5, 20, 50)
        df = pd.DataFrame( data = np.concatenate(
                                            (self.hmrm_domain.context_location_embedding,
                                            self.hmrm_domain.target_Location_embedding), 
                                            axis=1))

        if(weeplaces == "Yes" or weeplaces == "yes"):
            values = []
            for i in range(df.shape[0]):
                category = users_checkin[users_checkin["placeid_int"] == i]["category_new"].unique()[0]
                values.append(category)
            df["category"] = values
            category_to_int = {
                'Food': 2,
                'College & Education': 6,
                'Home / Work / Other': 1,
                'Home, Work, Others': 1,
                'Homes, Work, Others': 1,
                'Shops': 5,
                'Parks & Outdoors': 4,
                'Arts & Entertainment': 0,
                'Travel': 3,
                'Nightlife': 4,
                'Great Outdoors': 4,
                'Homes': 1,
                'Work': 1,
                'Others': 1,
                'Travel Spots': 0,
                'Colleges & Universities': 6,
                'Nightlife Spots': 4
            } ##Usar o que o claudio criou depois de dar merge
        
            df["category"] = df["category"].map(category_to_int)
        else:
            values = []
            for i in range(df.shape[0]):
                # category = users_checkin[users_checkin["placeid"] == i]["categoryid"].unique()[0]
                # values.append(category)
                values.append("Others")
            df["category"] = values
     

        self.file_loader.save_df_to_csv(df, output_filename)
        print(df.head(1))
        # df.to_csv(df, output_filename)

# '{"job":"hmrm_baseline", "users_checkin_filename":"gowalla_checkins.csv", "weeplaces?":"yes", "features_filename":"output.out"}'