import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="FL Security Dashboard", layout="wide")
st.title("Federated Learning Security Dashboard")
st.caption("All values come from saved experiment runs (results/master_results.csv). Nothing is hardcoded.")

@st.cache_data
def load_results():
    return pd.read_csv("results/master_results.csv")

m = load_results()
m = m[~m["security"]]                      # poisoning experiments, no security layer
last_round = m["round"].max()

tab1, tab2, tab3 = st.tabs(["Attack and defense results", "Security tests", "Run live experiment"])

with tab1:
    st.sidebar.header("Experiment settings")
    partition = st.sidebar.selectbox("Data distribution", ["iid", "noniid"])
    sub = m[m["partition"] == partition]
    attack = st.sidebar.selectbox("Attack", sorted(sub["attack"].unique()))
    counts = sorted(sub[sub["attack"] == attack]["num_malicious"].unique())
    n_mal = st.sidebar.selectbox("Malicious clients (out of 5)", counts)
    chosen = sub[(sub["attack"] == attack) & (sub["num_malicious"] == n_mal)]
    methods = sorted(chosen["aggregation"].unique())
    aggs = st.sidebar.multiselect("Aggregation methods", methods, default=methods)
    log_loss = st.sidebar.checkbox("Log scale for loss", value=True)

    base = sub[(sub["aggregation"] == "fedavg") & (sub["attack"] == "none")]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    bm = base.groupby("round")[["test_accuracy", "test_loss"]].mean()
    axes[0].plot(bm.index, bm["test_accuracy"], "k--", label="honest FedAvg")
    axes[1].plot(bm.index, bm["test_loss"], "k--", label="honest FedAvg")
    for a in aggs:
        g = chosen[chosen["aggregation"] == a].groupby("round")[["test_accuracy", "test_loss"]].agg(["mean", "std"])
        for ax, col in zip(axes, ["test_accuracy", "test_loss"]):
            mean, std = g[col]["mean"], g[col]["std"].fillna(0)
            ax.plot(g.index, mean, label=a)
            ax.fill_between(g.index, mean - std, mean + std, alpha=0.15)
    axes[0].set(xlabel="Round", ylabel="Test accuracy (%)", title="Accuracy")
    axes[1].set(xlabel="Round", ylabel="Test loss", title="Loss")
    if log_loss:
        axes[1].set_yscale("log")
    axes[0].legend(); axes[1].legend()
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader(f"Final round ({last_round}): {attack}, {n_mal} malicious, {partition}")
    base_final = base[base["round"] == last_round].set_index("seed")["test_accuracy"]
    f = chosen[(chosen["round"] == last_round) & (chosen["aggregation"].isin(aggs))].copy()
    f["accuracy_drop"] = f["seed"].map(base_final) - f["test_accuracy"]
    table = f.groupby("aggregation").agg(
        mean_accuracy=("test_accuracy", "mean"), std_accuracy=("test_accuracy", "std"),
        mean_drop=("accuracy_drop", "mean"), mean_loss=("test_loss", "mean")).round(2)
    st.dataframe(table)
    st.caption("Accuracy drop is measured against honest FedAvg with the same data distribution and seed. Mean and std are over 3 seeds.")

with tab2:
    for name, title in [("hmac_tests", "HMAC integrity tests"), ("replay_tests", "Replay and freshness tests")]:
        path = f"logs/{name}.csv"
        st.subheader(title)
        if os.path.exists(path):
            st.dataframe(pd.read_csv(path))
        else:
            st.info(f"{path} not found")
    st.caption("A poisoned update signed with a valid key is accepted: HMAC checks integrity, not honesty.")

with tab3:
    st.subheader("Run a live experiment")
    st.write("Runs the real federated-learning code with the settings below. Keep the rounds low: each round trains 5 clients on the CPU.")
    c1, c2, c3 = st.columns(3)
    with c1:
        l_part = st.selectbox("Data distribution", ["iid", "noniid"], key="l_part")
        l_agg = st.selectbox("Aggregation", ["fedavg", "median", "trimmed_mean"], key="l_agg")
        l_rounds = st.slider("Rounds", 1, 5, 3, key="l_rounds")
    with c2:
        l_attack = st.selectbox("Attack", ["none", "sign_flip", "random", "scaling"], key="l_attack")
        l_mal = st.selectbox("Malicious clients (out of 5)", [1, 2], key="l_mal")
        l_seed = st.selectbox("Seed", [42, 43, 44], key="l_seed")
    with c3:
        l_sec = st.checkbox("Enable HMAC and replay checks", value=False, key="l_sec")
        l_log = st.checkbox("Log scale for loss", value=False, key="l_log")

    n_mal = 0 if l_attack == "none" else l_mal
    st.caption(f"Malicious clients used: {n_mal} (an attack of 'none' always uses 0).")

    if st.button("Run experiment"):
        from src.experiment import run_experiment
        with st.spinner("Training... this can take a few minutes. Please wait."):
            rows = run_experiment(seed=l_seed, rounds=l_rounds, attack=l_attack,
                                  num_malicious=n_mal, partition=l_part,
                                  aggregation=l_agg, security=l_sec)
        run_df = pd.DataFrame(rows)
        os.makedirs("results", exist_ok=True)
        hist_path = "results/live_runs.csv"
        run_df.to_csv(hist_path, mode="a", header=not os.path.exists(hist_path), index=False)
        st.session_state["live"] = {"df": run_df, "cfg": {
            "partition": l_part, "aggregation": l_agg, "attack": l_attack,
            "mal": n_mal, "seed": l_seed, "rounds": l_rounds, "security": l_sec, "log": l_log}}

    if "live" in st.session_state:
        res = st.session_state["live"]
        run_df, cfg = res["df"], res["cfg"]
        st.markdown(f"**Last run:** {cfg['partition']}, {cfg['aggregation']}, attack = {cfg['attack']}, "
                    f"{cfg['mal']} malicious, seed {cfg['seed']}, {cfg['rounds']} rounds, "
                    f"security {'on' if cfg['security'] else 'off'}")

        hb = m[(m["partition"] == cfg["partition"]) & (m["aggregation"] == "fedavg") &
               (m["attack"] == "none") & (m["seed"] == cfg["seed"]) &
               (m["round"] <= cfg["rounds"])].sort_values("round")

        fig2, ax2 = plt.subplots(1, 2, figsize=(10, 3.8))
        for ax, col, title in [(ax2[0], "test_accuracy", "Accuracy (%)"), (ax2[1], "test_loss", "Loss")]:
            ax.plot(hb["round"], hb[col], "k--", label="saved honest FedAvg (same seed)")
            ax.plot(run_df["round"], run_df[col], marker="o", label="live run")
            ax.set(xlabel="Round", ylabel=title, title=title)
            ax.legend()
        if cfg["log"]:
            ax2[1].set_yscale("log")
        plt.tight_layout()
        st.pyplot(fig2)

        final = run_df.iloc[-1]
        ref = hb[hb["round"] == cfg["rounds"]]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Final accuracy", f"{final['test_accuracy']:.2f}%")
        k2.metric("Final loss", f"{final['test_loss']:.4f}")
        k3.metric("Runtime", f"{final['runtime_sec']:.0f} s")
        if len(ref) > 0:
            k4.metric("Drop vs honest FedAvg", f"{ref.iloc[0]['test_accuracy'] - final['test_accuracy']:.2f}")

        if cfg["security"]:
            st.subheader("Security layer")
            st.write(f"Tag time: {run_df['tag_time_sec'].sum():.3f} s | "
                     f"Verification time: {run_df['verify_time_sec'].sum():.3f} s | "
                     f"Data sent: {int(run_df['msg_bytes'].sum()):,} bytes | "
                     f"Messages rejected: {int(run_df['rejected'].sum())}")
            log_path = (f"logs/secure_run_{cfg['partition']}_{cfg['aggregation']}_"
                        f"{cfg['attack']}_m{cfg['mal']}_seed{cfg['seed']}.csv")
            if os.path.exists(log_path):
                st.dataframe(pd.read_csv(log_path))

    if os.path.exists("results/live_runs.csv"):
        with st.expander("History of live runs"):
            hist = pd.read_csv("results/live_runs.csv")
            keys = ["partition", "aggregation", "attack", "num_malicious", "seed", "security"]
            st.dataframe(hist.groupby(keys, as_index=False).tail(1)[keys + ["round", "test_accuracy", "test_loss", "runtime_sec"]])