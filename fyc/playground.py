import desc.io
import desc.plotting as dplot
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from pprint import pprint

if __name__ == "__main__":
    ir = desc.io.InputReader()
    input_list = ir.parse_inputs(fname="DSHAPE_CURRENT")
    pprint(input_list)
