import os
import sys
import warnings

warnings.filterwarnings("ignore")

from hmrm.configuration.data import OutputDirectory, InputDirectory
from hmrm.base.hmrmBaseline import HmrmBaseline

if __name__ == "__main__":
    base_dir = "/media/gegen07/Expansion/data/mestrado/checkins_data/checkins_local"
    file = os.path.join(base_dir, "checkins_local_Alabama.csv")

    weight = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    K = [7]
    embedding_size = [10, 20, 30, 40, 50, 100, 150, 200, 250, 300]

    # for w in weight:
    #     for k in K:
    #         for e in embedding_size:
    #             out = OutputDirectory(f"features-{w}-{k}-{e}.csv")
    #             hmrm = HmrmBaseline(file, w, k, e)
    #             df = hmrm.start()
    #             print("Weight: ", w, " K: ", k, " Embedding Size: ", e)
    #             df.to_csv(out.outfile, index=False)

    hmrm = HmrmBaseline(file, 0.1, 7, 50)
    df = hmrm.start()
    df.to_csv(OutputDirectory.outfile, index=False)

