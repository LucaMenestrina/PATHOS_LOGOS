import pandas as pd
import numpy as np
from datetime import datetime
import os

import torch

from pykeen.triples import TriplesFactory
from pykeen.predict import predict_target
from pykeen.utils import set_random_seed

random_seed = 12345
split_ratio = [0.8, 0.1, 0.1]
directory = "predictions"

set_random_seed(random_seed)

print(f"Predictions started ({datetime.now().strftime('%Y/%m/%d, %H:%M:%S')})")

if not(os.path.exists(directory) and os.path.isdir(directory)):
    os.makedirs(directory)

# Load dataset
PATHOS = pd.read_csv("../PATHOS/data/KG.tsv.gz", sep="\t", index_col=0)
triples_factory = TriplesFactory.from_path_binary("HPO/triples_factory")
training_triples, testing_triples, validation_triples = TriplesFactory.from_path_binary("HPO/triples_factory/training_triples"), TriplesFactory.from_path_binary("HPO/triples_factory/testing_triples"), TriplesFactory.from_path_binary("HPO/triples_factory/validation_triples")

pd.DataFrame.from_dict(
    triples_factory.entity_to_id, orient="index", columns=["id"]
    ).reset_index(
        ).set_index("id").rename(columns={"index":"label"}
                                 ).merge(
                                     pd.concat([
                                         PATHOS[["subject", "subjectName", "subjectType"]].rename(columns={"subject":"label", "subjectName":"name", "subjectType":"type"}),
                                         PATHOS[["object", "objectName", "objectType"]].rename(columns={"object":"label", "objectName":"name", "objectType":"type"})
                                         ]).drop_duplicates("label").reset_index(drop=True), on="label", how="left").to_csv("trained/entities.tsv", sep="\t")
pd.DataFrame.from_dict(triples_factory.relation_to_id, orient="index", columns=["id"]).reset_index().rename(columns={"index":"label"})[["id", "label"]].to_csv(f"trained/relations.tsv", sep="\t", index=False)


LOGOS = torch.load(f"trained/model.pkl")

# Save embeddings
np.save("trained/entities_embedding.npy", LOGOS.entity_representations[0]().cpu().detach().numpy())
np.save("trained/relations_embedding.npy", LOGOS.relation_representations[0]().cpu().detach().numpy())

## Predict
# Drug Repurposing
disease = "AD"
disease_id = "MONDO:0004975"
predicted = predict_target(
    model=LOGOS,
    tail=disease_id,
    relation="indication",
    triples_factory=triples_factory,
)
predicted = predicted.add_membership_columns(training=training_triples, validation=validation_triples, testing=testing_triples)
predicted.df.sort_values(by="score", ascending=False).reset_index(drop=True).to_csv(f"{directory}/{disease.replace(' ', '_')}_indications.tsv", sep="\t")

# has phenotype
disease = "HD"
disaese_id = "MONDO:0007739"
predicted = predict_target(
    model=LOGOS,
    head=disease_id,
    relation="has_phenotype",
    triples_factory=triples_factory,
)
predicted = predicted.add_membership_columns(training=training_triples, validation=validation_triples, testing=testing_triples)
predicted.df.sort_values(by="score", ascending=False).reset_index(drop=True).to_csv(f"{directory}/{disease.replace(' ', '_')}_has_phenotype.tsv", sep="\t")

# related protein
disease = "MS"
disease_id = "MONDO:0005301"

predicted = predict_target(
    model=LOGOS,
    tail=disease_id,
    relation="related_to_disease",
    triples_factory=triples_factory,
)
predicted = predicted.add_membership_columns(training=training_triples, validation=validation_triples, testing=testing_triples)
predicted.df.sort_values(by="score", ascending=False).reset_index(drop=True).to_csv(f"{directory}/{disease.replace(' ', '_')}_related_genes.tsv", sep="\t")

print(f"Predictions ended ({datetime.now().strftime('%Y/%m/%d, %H:%M:%S')})")
