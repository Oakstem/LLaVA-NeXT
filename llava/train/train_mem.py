# Fix NUMA affinity issues in SLURM/distributed environments - MUST BE FIRST
import os
os.environ["ACCELERATE_DISABLE_NUMA"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from llava.train.train import train

if __name__ == "__main__":
    train()
