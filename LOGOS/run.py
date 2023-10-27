# this is a file ready to be launched on a cluster
# run it with: python run.py -m "model_name" -r replica_number
# e.g. python run.py -m "RotatE" -r 1

import pandas as pd
import numpy as np
import json
import os
import argparse
import logging

from pykeen.utils import set_random_seed
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline_from_config

# Initialize arg parser
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="Model name")
parser.add_argument("-r", "--replica", help="Replica number")

args = parser.parse_args()

model = args.model
replica = int(args.replica)

if not(os.path.exists(f"{model}/{replica}") and os.path.isdir(f"{model}/{replica}")):
        os.makedirs(f"{model}/{replica}")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"{model}/{replica}/log"),
        logging.StreamHandler()
    ]
)

random_seed = replica
set_random_seed(random_seed)

logging.info("Loading dataset")
df = pd.read_csv("../../PATHOS/data/KG_slim.tsv.gz", sep="\t", compression="gzip")

training_triples, testing_triples, validation_triples = TriplesFactory.from_path_binary("../HPO/triples_factory/training_triples"), TriplesFactory.from_path_binary("../HPO/triples_factory/testing_triples"), TriplesFactory.from_path_binary("../HPO/triples_factory/validation_triples")

logging.info("Loading pipeline config")
with open("../HPO/best_pipeline/pipeline_config.json") as infile:
    config = dict(json.load(infile))

config["pipeline"]["training"] = training_triples
config["pipeline"]["validation"] = validation_triples
config["pipeline"]["testing"] = testing_triples
config["pipeline"]["random_seed"] = replica
config["pipeline"]["model_kwargs"]["interaction"] = model

logging.info("Launching pipeline")
pipeline_results = pipeline_from_config(
        config=config,
        use_testing_data=True,
        move_to_cpu=False,
        save_training=False
    )
logging.info("Pipeline completed")

logging.info("Saving pipeline")
pipeline_results.save_to_directory(f"{model}/{replica}")

logging.info("Saving embeddings")
LOGOS = pipeline_results.model
np.save(f"{model}/{replica}/entities_embedding.npy", LOGOS.entity_representations[0]().cpu().detach().numpy())
np.save(f"{model}/{replica}/relations_embedding.npy", LOGOS.relation_representations[0]().cpu().detach().numpy())

logging.info("End")
