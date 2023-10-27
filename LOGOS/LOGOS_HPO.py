import pandas as pd
from datetime import datetime
import os

from pykeen.triples import TriplesFactory
from pykeen.models import NodePiece
from pykeen.hpo import hpo_pipeline
from pykeen.utils import set_random_seed

random_seed = 12345
split_ratio = [0.8, 0.1, 0.1]

set_random_seed(random_seed)
if not(os.path.exists("HPO") and os.path.isdir("HPO")):
    os.makedirs("HPO")

print(f"Load dataset ({datetime.now().strftime('%Y/%m/%d, %H:%M:%S')})")

df = pd.read_csv("../PATHOS/data/KG_slim.tsv.gz", sep="\t", compression="gzip")
triples_factory = TriplesFactory.from_labeled_triples(
    triples=df.values,
    create_inverse_triples=True
)
os.makedirs("HPO/triples_factory")
triples_factory.to_path_binary("HPO/triples_factory/")
training_triples, testing_triples, validation_triples = triples_factory.split(ratios=split_ratio, random_state=random_seed)
training_triples.to_path_binary("HPO/triples_factory/training_triples")
testing_triples.to_path_binary("HPO/triples_factory/testing_triples")
validation_triples.to_path_binary("HPO/triples_factory/validation_triples")

print(f"Start HPO ({datetime.now().strftime('%Y/%m/%d, %H:%M:%S')})")

hpo_pipeline_result = hpo_pipeline(
    study_name="PATHOS_LOGOS_HPO",
    # n_trials=1,
    timeout= 60*60*24*7, # optimization time in seconds (here ss*mm*hh*dd)
    training=training_triples,
    validation=validation_triples,
    testing=testing_triples,
    model=NodePiece,
    model_kwargs=dict(
        aggregation="mlp",
        random_seed=random_seed,
        interaction="rotate",
        relation_initializer="init_phases",
        relation_constrainer="complex_normalize",
        entity_initializer="xavier_uniform_",
        tokenizers=["AnchorTokenizer", "RelationTokenizer"],
        num_tokens=(20, 5),
        tokenizers_kwargs=[
                    dict(
                        selection="MixtureAnchorSelection",
                        selection_kwargs=dict(
                            selections=["degree", "random"],  # ["degree", "pagerank", "random"],  # pagerank throws an error when checks the adjacency matrix
                            ratios=[0.8, 0.2],  # [0.4, 0.4, 0.2],
                            num_anchors=10000,
                        ),
                        searcher="ScipySparse",
                        searcher_kwargs=dict(max_iter=100)
                    ),
                    dict(),  # empty dict for the RelationTokenizer - it doesn't need any kwargs
                ],
    ),
    model_kwargs_ranges=dict(
        embedding_dim=dict(type=int, scale="power", base=2, low=6, high=9),
    ),
    optimizer="Adam",
    optimizer_kwargs=dict(
        lr=0.0001,
    ),
    loss="nssa",
    loss_kwargs_ranges=dict(
        margin=dict(type="categorical", choices=[50, 100]),
    ),
    negative_sampler="bernoulli",
    negative_sampler_kwargs_ranges=dict(
        num_negs_per_pos=dict(type="categorical", choices=[5, 10, 100]),
    ),
    training_loop="slcwa",
    training_kwargs=dict(
        num_epochs=200,
    ),
    training_kwargs_ranges=dict(
        batch_size=dict(type="categorical", choices=[256, 512, 1024]),
    ),
    evaluator="rankbased",
    evaluator_kwargs=dict(
        filtered=True,    
        metrics=["adjusted_mean_rank", "mean_rank", "mean_reciprocal_rank", "adjusted_mean_reciprocal_rank", "hits@k"],
    ),
    evaluation_kwargs=dict(
        batch_size=64,
        # ks=[1, 5, 10]
    ),
    stopper="early",
    stopper_kwargs=dict(
        metric="adjusted_mean_reciprocal_rank",
        larger_is_better=True,
        frequency=10,
        patience=3,
        relative_delta=0.001,
    ),
    metric="adjusted_mean_reciprocal_rank",
    direction="maximize",
    sampler="tpe",
    sampler_kwargs=dict(
        seed=random_seed,
    ),
    device="cuda",
    save_model_directory="HPO/models"
)

hpo_pipeline_result.save_to_directory("HPO")

print(f"HPO finished ({datetime.now().strftime('%Y/%m/%d, %H:%M:%S')})")