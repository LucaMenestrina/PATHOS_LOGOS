import pandas as pd
import json
import os
import shutil

models = ["RotatE", "ComplEx", "TransE", "DistMult"]
n_replicas = 5

if not(os.path.exists("comparison") and os.path.isdir("comparison")):
        os.makedirs("comparison")

os.system("cd comparison")
for replica in range(1, n_replicas+1):
     for model in models:
           os.system(f"python run.py -m {model} -r {replica}")
os.system("cd ..")

if __name__ == "__main__":
    metrics = pd.DataFrame(columns=["model", "replica", "adjusted_inverse_harmonic_mean_rank", "adjusted_hits_at_k", "inverse_harmonic_mean_rank", "hits_at_10", "hits_at_5", "hits_at_1"])
    for model in models:
        model_metrics = pd.DataFrame(columns=["model", "replica", "adjusted_inverse_harmonic_mean_rank", "adjusted_hits_at_k", "inverse_harmonic_mean_rank", "hits_at_10", "hits_at_5", "hits_at_1"])
        replicas = os.listdir(f"comparison/{model}")
        for replica in replicas:
            replica = int(replica)
            with open(f"comparison/{model}/{replica}/results.json", "r") as input:
                results = json.load(input)
            replica_metrics = pd.DataFrame(results["metrics"]["both"]["realistic"], index=[replica])[["adjusted_inverse_harmonic_mean_rank", "adjusted_hits_at_k", "inverse_harmonic_mean_rank", "hits_at_10", "hits_at_5", "hits_at_1"]]
            replica_metrics["model"] = model
            replica_metrics["replica"] = replica
            model_metrics = pd.concat([model_metrics, replica_metrics])
        metrics = pd.concat([metrics, model_metrics])
    metrics = metrics.reset_index(drop=True)
    metrics.to_csv("comparison/metrics.tsv", sep="\t")
    metrics_stats = metrics.groupby("model").agg(["mean", "std", "var", "sem", "min", "max"])[["adjusted_inverse_harmonic_mean_rank", "adjusted_hits_at_k", "inverse_harmonic_mean_rank", "hits_at_10", "hits_at_5", "hits_at_1"]]
    metrics_stats.to_csv("comparison/metrics_stats.tsv", sep="\t")
    # Save best model
    if not(os.path.exists("trained") and os.path.isdir("trained")):
        os.makedirs("trained")
    best_model, best_replica = metrics.iloc[metrics["adjusted_inverse_harmonic_mean_rank"].argmax()][["model", "replica"]].values
    shutil.copy2(f"comparison/{best_model}/{best_replica}/trained_model.pkl", "trained/model.pkl")
    # Save pipeline config
    with open("HPO/best_pipeline/pipeline_config.json") as infile:
        config = dict(json.load(infile))
    config["pipeline"]["model"] = best_model
    with open("trained/pipeline_config.json", "w") as outfile:
        json.dump(config, outfile)
