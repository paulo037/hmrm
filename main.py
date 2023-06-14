import sys
import ast
import os
import warnings
warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from job.hmrm_job import HmrmBaseline
from foundation.configuration.input import Input

def start_input(args):
    Input().set_inputs(args)


def start_job(args):

    start_input(args)
    job_name = Input.get_instance().inputs['job']
    print(job_name)

    job = HmrmBaseline()
    job.start()

if __name__ == "__main__":
    try:

        args = ast.literal_eval(sys.argv[1])
        print(args)
        start_job(args)

    except Exception as e:
        raise e