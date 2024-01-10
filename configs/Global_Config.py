import torch

run_in_colab = False
run_in_notebook = False
run_in_slurm = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
step = 0
BASE_PATH = ''
