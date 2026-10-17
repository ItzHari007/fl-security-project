import os
import pandas as pd
from src.experiment import run_experiment

def run_grid(partitions, aggregations, scenarios, seeds, out_csv, rounds=10, trim_k=1):
    """scenarios is a list of (attack, num_malicious) pairs, e.g. [("none", 0), ("sign_flip", 1)]."""
    done = pd.read_csv(out_csv) if os.path.exists(out_csv) else pd.DataFrame()

    for part in partitions:
        for agg in aggregations:
            for attack, m in scenarios:
                for seed in seeds:
                    label = f"{part} | {agg} | {attack} | malicious={m} | seed={seed}"
                    if len(done) > 0:
                        already = ((done["partition"] == part) &
                                   (done["aggregation"] == agg) &
                                   (done["attack"] == attack) &
                                   (done["num_malicious"] == m) &
                                   (done["seed"] == seed)).any()
                        if already:
                            print(f"Skipping (already done): {label}")
                            continue
                    print(f"Running: {label} ...")
                    rows = run_experiment(seed=seed, rounds=rounds, attack=attack,
                                          num_malicious=m, partition=part,
                                          aggregation=agg, trim_k=trim_k)
                    done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
                    done.to_csv(out_csv, index=False)   # save after every run
    return done
