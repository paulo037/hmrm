import os
from dataclasses import dataclass


@dataclass
class OutputDirectory:
    outdir: str = "out"
    outfile: str = os.path.join(outdir, "features.csv")

    def __init__(self, file=None):
        self.outdir = "out"
        if file is None:
            self.outfile = os.path.join(self.outdir, "features.csv")
        else:
            self.outfile = os.path.join(self.outdir, file)

    def __post_init__(self):
        os.makedirs(self.outdir, exist_ok=True)


@dataclass
class InputDirectory:
    indir: str
    infile: str

    def __init__(self, file=None):
        self.indir = os.path.basename(os.path.dirname(file))
        self.infile = file
