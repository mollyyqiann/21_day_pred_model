"""Model comparison: multiple feature sets for GBC, vanilla LSTM baseline,
and a stock-modified hybrid (LSTM over market-adjusted sequences + MLP over
regime-shift aggregates).

The panel used for training is the v2 panel (historically-calm SP500, ~319
tickers, 240k rows) — it has enough data to support sequence models. The v3
universe (calm->vol->calm, 6x recency) is used later at inference time to
scope where we actually surface predictions.

Feature sets for GBC ablation:
  A_base      - classical technicals only (RSI, MACD, BB, ATR, volume)
  B_plus_mom  - A + multi-horizon momentum + moving-average gaps
  C_plus_reg  - B + volatility, beta residuals, regime-shift, SPY context (FULL)

Sequence models:
  vanilla_lstm - LSTM on 5-channel raw sequence (logret, logvol_z,
                 resid_vs_spy, range_pct, rv20) of 30 days -> 1-dim logit.
  hybrid_lstm  - same LSTM branch concatenated with normalized aggregate
                 features (the 'C_plus_reg' set) through an MLP head.

Metric focus: test-set AUC and PR-AUC lift over base rate. These are the
honest comparison metrics for a 1% base-rate problem.

Outputs:
  output/burst_models_compare.json
  output/burst_today_v3.csv  (scored for the v3 universe using the winning model)
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
torch.manual_seed(0)
np.random.seed(0)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "output"
MODELS = ROOT / "models"
OUT.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True)

SEQ_LEN = 30
DEVICE = "cpu"


# ---------- feature subsets ----------
A_BASE = ["rsi_14", "macd", "macd_sig", "macd_hist", "bb_z20",
          "atr_pct", "range_pct", "vol_z", "vol_5d"]
B_MOM = A_BASE + ["ret_1d", "ret_5d", "ret_10d", "ret_20d", "ret_60d",
                  "gap_ma50", "gap_ma200", "pos_52w", "pct_from_52w_high"]
C_FULL = B_MOM + ["rv_10", "rv_20", "rv_60", "rv_120",
                  "rv_ratio_20_60", "rv_ratio_60_120", "rv_ratio_vs_hist",
                  "max_ret_20d", "max_ret_60d",
                  "num_3pct_days_60d", "num_5pct_days_60d", "max_3d_avg_60d",
                  "beta_60", "resid_1d", "resid_5d", "resid_20d",
                  "spy_ret_5d", "spy_ret_20d", "spy_rv_20"]

FSETS = {"A_base": A_BASE, "B_plus_mom": B_MOM, "C_plus_reg": C_FULL}


# ---------- metric helper ----------

def evaluate(y_true, p) -> dict:
    base = float(np.mean(y_true)) if len(y_true) else float("nan")
    auc = roc_auc_score(y_true, p) if len(np.unique(y_true)) > 1 else float("nan")
    ap = average_precision_score(y_true, p) if len(np.unique(y_true)) > 1 else float("nan")
    ll = log_loss(y_true, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])
    bs = brier_score_loss(y_true, p)
    return {"n": int(len(y_true)), "pos": int(y_true.sum()), "base": base,
            "auc": float(auc), "ap": float(ap),
            "ap_lift": float(ap / base) if base > 0 else float("nan"),
            "log_loss": float(ll), "brier": float(bs)}


# ---------- sequence builder ----------

SEQ_CHANS = ["ret_1d", "vol_z", "resid_1d", "range_pct", "rv_20"]


def build_sequences(panel: pd.DataFrame, feat_cols_seq, seq_len: int = SEQ_LEN):
    """Build (N, seq_len, C) arrays per ticker. Skip the first seq_len-1 rows."""
    X_list, agg_list, y_list, idx_list = [], [], [], []
    for t, g in panel.groupby("ticker", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        if len(g) < seq_len:
            continue
        X = g[feat_cols_seq].fillna(0.0).values.astype(np.float32)
        # aggregate features (C_FULL) at the *last* step of each window
        A = g[C_FULL].fillna(0.0).values.astype(np.float32)
        y = g["y"].values
        for i in range(seq_len - 1, len(g)):
            X_list.append(X[i - seq_len + 1 : i + 1])
            agg_list.append(A[i])
            y_list.append(y[i])
            idx_list.append(g.index[i] if False else (t, g["date"].iloc[i]))
    if not X_list:
        return None
    return (np.stack(X_list), np.stack(agg_list),
            np.array(y_list, dtype=np.int8), idx_list)


# ---------- LSTM models ----------

class VanillaLSTM(nn.Module):
    def __init__(self, in_ch: int, hidden: int = 64, drop: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(in_ch, hidden, num_layers=1, batch_first=True)
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x, agg=None):
        h, _ = self.lstm(x)
        h = h[:, -1, :]
        return self.fc(self.drop(h)).squeeze(-1)


class HybridLSTM(nn.Module):
    """Sequence branch + regime-aggregate branch, concatenated then MLP."""
    def __init__(self, in_ch: int, agg_dim: int, hidden: int = 48, drop: float = 0.25):
        super().__init__()
        self.lstm = nn.LSTM(in_ch, hidden, num_layers=1, batch_first=True)
        self.seq_ln = nn.LayerNorm(hidden)
        self.agg_mlp = nn.Sequential(
            nn.Linear(agg_dim, 32), nn.ReLU(), nn.Dropout(drop),
            nn.LayerNorm(32),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden + 32, 32), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(32, 1),
        )

    def forward(self, x, agg):
        h, _ = self.lstm(x); h = self.seq_ln(h[:, -1, :])
        a = self.agg_mlp(agg)
        return self.head(torch.cat([h, a], dim=-1)).squeeze(-1)


def train_torch(model, Xtr, agg_tr, ytr, Xv, agg_v, yv, pos_weight,
                epochs: int = 18, batch: int = 512, lr: float = 1e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    n = len(Xtr); best_ap = -1; best_state = None; patience = 0
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(n)
        ep_loss = 0
        for s in range(0, n, batch):
            idx = perm[s:s+batch]
            xb = torch.from_numpy(Xtr[idx])
            ab = torch.from_numpy(agg_tr[idx])
            yb = torch.from_numpy(ytr[idx]).float()
            opt.zero_grad()
            logits = model(xb, ab)
            loss = loss_fn(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item() * len(idx)
        # val AP
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(torch.from_numpy(Xv),
                                      torch.from_numpy(agg_v))).numpy()
        ap = average_precision_score(yv, pv) if yv.sum() > 0 else 0.0
        print(f"   ep{ep+1:2d}  loss={ep_loss/n:.4f}  val_AP={ap:.4f}")
        if ap > best_ap:
            best_ap = ap; best_state = {k: v.clone() for k, v in model.state_dict().items()}; patience = 0
        else:
            patience += 1
            if patience >= 4:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def main():
    meta = json.loads((DATA / "burst_meta_v2.json").read_text())
    panel = pd.read_csv(DATA / "burst_panel_v2.csv", parse_dates=["date"])
    print(f"[models] panel rows: {len(panel):,}")

    # drop rows whose labels are unknown (y==-1) for train/val/test purposes
    lab = panel[panel["y"] >= 0].copy()
    live = panel[panel["y"] == -1].copy()

    # drop rows missing any of the union of feature columns
    needed = list(set(C_FULL + SEQ_CHANS))
    lab = lab.dropna(subset=needed).reset_index(drop=True)
    print(f"[models] labelled rows (post-NA): {len(lab):,}  base-rate: {lab['y'].mean():.4%}")

    # chronological split
    dates = np.sort(lab["date"].unique())
    n = len(dates)
    d1 = dates[int(0.70 * n)]; d2 = dates[int(0.85 * n)]
    tr = lab[lab["date"] < d1]; va = lab[(lab["date"] >= d1) & (lab["date"] < d2)]; te = lab[lab["date"] >= d2]
    print(f"[models] train/val/test sizes: {len(tr):,} / {len(va):,} / {len(te):,}")

    results = {}

    # ========== GBC ablations ==========
    for name, cols in FSETS.items():
        print(f"\n[GBC:{name}] features: {len(cols)}")
        gbc = GradientBoostingClassifier(n_estimators=300, max_depth=3,
                                         learning_rate=0.05, subsample=0.8,
                                         random_state=42)
        gbc.fit(tr[cols].values, tr["y"].values)
        p_tr = gbc.predict_proba(tr[cols].values)[:, 1]
        p_va = gbc.predict_proba(va[cols].values)[:, 1]
        p_te = gbc.predict_proba(te[cols].values)[:, 1]
        r = {"train": evaluate(tr["y"].values, p_tr),
             "val":   evaluate(va["y"].values, p_va),
             "test":  evaluate(te["y"].values, p_te),
             "n_features": len(cols)}
        print(f"   test: AUC={r['test']['auc']:.3f}  AP_lift={r['test']['ap_lift']:.1f}x"
              f"  log_loss={r['test']['log_loss']:.4f}")
        results[f"GBC_{name}"] = r
        if name == "C_plus_reg":
            joblib.dump({"gbc": gbc, "feat_cols": cols}, MODELS / "burst_gbc_v2.joblib")

    # ========== sequences for LSTM / hybrid ==========
    print("\n[seq] building sequences ...")
    seq_lab = build_sequences(lab, SEQ_CHANS, seq_len=SEQ_LEN)
    if seq_lab is None:
        raise RuntimeError("no sequences")
    X_all, A_all, y_all, idx_all = seq_lab
    print(f"[seq] total sequences: {len(X_all):,}  pos: {int(y_all.sum()):,}")

    # reconstruct split from idx dates
    seq_dates = pd.to_datetime([d for (_, d) in idx_all])
    d1_ts = pd.Timestamp(d1); d2_ts = pd.Timestamp(d2)
    mask_tr = np.asarray(seq_dates < d1_ts)
    mask_va = np.asarray((seq_dates >= d1_ts) & (seq_dates < d2_ts))
    mask_te = np.asarray(seq_dates >= d2_ts)

    # per-channel standardization fit on train only
    tr_X = X_all[mask_tr]
    mu = tr_X.reshape(-1, X_all.shape[-1]).mean(0)
    sd = tr_X.reshape(-1, X_all.shape[-1]).std(0) + 1e-6
    X_all_s = (X_all - mu) / sd

    # aggregate standardization
    agg_scaler = StandardScaler().fit(A_all[mask_tr])
    A_all_s = agg_scaler.transform(A_all).astype(np.float32)

    ytr = y_all[mask_tr]; yv = y_all[mask_va]; yte = y_all[mask_te]
    pos_weight = float((ytr == 0).sum() / max(1, (ytr == 1).sum()))
    print(f"[seq] split sizes: tr={mask_tr.sum()} va={mask_va.sum()} te={mask_te.sum()}"
          f"  pos_weight={pos_weight:.1f}")

    # ---- vanilla LSTM ----
    print("\n[LSTM-vanilla] training ...")
    m1 = VanillaLSTM(in_ch=X_all.shape[-1], hidden=64, drop=0.2)
    m1 = train_torch(m1, X_all_s[mask_tr].astype(np.float32), A_all_s[mask_tr],
                     ytr, X_all_s[mask_va].astype(np.float32), A_all_s[mask_va], yv,
                     pos_weight=pos_weight)
    m1.eval()
    with torch.no_grad():
        p_te = torch.sigmoid(m1(torch.from_numpy(X_all_s[mask_te].astype(np.float32)),
                                 torch.from_numpy(A_all_s[mask_te]))).numpy()
    results["LSTM_vanilla"] = {
        "test": evaluate(yte, p_te), "n_features": X_all.shape[-1]
    }
    print(f"   test: AUC={results['LSTM_vanilla']['test']['auc']:.3f}  "
          f"AP_lift={results['LSTM_vanilla']['test']['ap_lift']:.1f}x")
    torch.save(m1.state_dict(), MODELS / "burst_lstm_vanilla.pt")

    # ---- hybrid LSTM+MLP ----
    print("\n[Hybrid] training ...")
    m2 = HybridLSTM(in_ch=X_all.shape[-1], agg_dim=A_all.shape[-1], hidden=48, drop=0.25)
    m2 = train_torch(m2, X_all_s[mask_tr].astype(np.float32), A_all_s[mask_tr],
                     ytr, X_all_s[mask_va].astype(np.float32), A_all_s[mask_va], yv,
                     pos_weight=pos_weight, epochs=22)
    m2.eval()
    with torch.no_grad():
        p_te_hy = torch.sigmoid(m2(torch.from_numpy(X_all_s[mask_te].astype(np.float32)),
                                     torch.from_numpy(A_all_s[mask_te]))).numpy()
    results["LSTM_hybrid"] = {
        "test": evaluate(yte, p_te_hy), "n_features_seq": X_all.shape[-1],
        "n_features_agg": A_all.shape[-1]
    }
    print(f"   test: AUC={results['LSTM_hybrid']['test']['auc']:.3f}  "
          f"AP_lift={results['LSTM_hybrid']['test']['ap_lift']:.1f}x")
    torch.save(m2.state_dict(), MODELS / "burst_lstm_hybrid.pt")

    # ========== comparison table ==========
    print("\n========== test-set comparison ==========")
    print(f"{'model':<22}  {'AUC':>6}  {'AP':>6}  {'AP_lift':>8}  {'log_loss':>9}")
    comp_rows = []
    for name, r in results.items():
        t = r["test"]
        print(f"{name:<22}  {t['auc']:>6.3f}  {t['ap']:>6.3f}  "
              f"{t['ap_lift']:>7.2f}x  {t['log_loss']:>9.4f}")
        comp_rows.append({"model": name, **t})

    (OUT / "burst_models_compare.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame(comp_rows).to_csv(OUT / "burst_models_compare.csv", index=False)

    # ========== score v3 universe today with WINNING model ==========
    winner = max(results.items(), key=lambda kv: kv[1]["test"]["ap"])
    print(f"\n[winner] {winner[0]}  (test AP = {winner[1]['test']['ap']:.3f})")

    # v3 universe
    uni_v3 = pd.read_csv(DATA / "burst_universe_v3.csv")
    print(f"[score] v3 universe size: {len(uni_v3)}")

    # For each v3 ticker, take the latest fully-featured row from the panel
    # (includes live rows where y == -1).
    scored = panel.dropna(subset=needed).copy()
    latest = (scored[scored["ticker"].isin(uni_v3["ticker"])]
              .sort_values(["ticker", "date"])
              .groupby("ticker").tail(1).copy())

    # score with winning model
    wname = winner[0]
    if wname.startswith("GBC_"):
        cols = FSETS[wname[4:]]
        if wname == "GBC_C_plus_reg":
            art = joblib.load(MODELS / "burst_gbc_v2.joblib")
            latest["prob"] = art["gbc"].predict_proba(latest[cols].values)[:, 1]
        else:
            # retrain quickly on that subset to get weights (cheap)
            gbc = GradientBoostingClassifier(n_estimators=300, max_depth=3,
                                             learning_rate=0.05, subsample=0.8,
                                             random_state=42)
            gbc.fit(tr[cols].values, tr["y"].values)
            latest["prob"] = gbc.predict_proba(latest[cols].values)[:, 1]
    else:
        # sequence model: need last SEQ_LEN rows per ticker
        rows = []
        for _, r in latest.iterrows():
            t = r["ticker"]; d = r["date"]
            hist = panel[(panel["ticker"] == t) & (panel["date"] <= d)].dropna(subset=needed)
            if len(hist) < SEQ_LEN:
                rows.append(float("nan")); continue
            x = hist[SEQ_CHANS].tail(SEQ_LEN).values.astype(np.float32)
            x = (x - mu) / sd
            a = agg_scaler.transform(hist[C_FULL].tail(1).values).astype(np.float32)
            with torch.no_grad():
                model = m1 if wname == "LSTM_vanilla" else m2
                p = torch.sigmoid(model(torch.from_numpy(x[None, ...]),
                                        torch.from_numpy(a))).item()
            rows.append(p)
        latest["prob"] = rows

    # join universe metadata
    latest = latest.merge(
        uni_v3[["ticker", "baseline_rv20", "rv_60_now", "n_episodes_qualifying",
                "mre_start", "mre_length", "mre_days_since", "mre_peak_rv", "sector"]],
        on="ticker", how="left")

    # rank
    latest = latest.sort_values("prob", ascending=False).reset_index(drop=True)
    latest["base_rate_test"] = results[winner[0]]["test"]["base"]
    latest["lift"] = latest["prob"] / latest["base_rate_test"]
    out_cols = ["ticker", "date", "close", "prob", "lift",
                "baseline_rv20", "rv_60_now", "n_episodes_qualifying",
                "mre_start", "mre_length", "mre_days_since", "mre_peak_rv",
                "sector", "ret_5d", "rsi_14", "rv_ratio_vs_hist",
                "num_3pct_days_60d", "resid_5d", "vol_z"]
    out_cols = [c for c in out_cols if c in latest.columns]
    latest[out_cols].to_csv(OUT / "burst_today_v3.csv", index=False)
    print(f"[score] wrote output/burst_today_v3.csv")
    print(latest[out_cols].head(20).to_string(index=False))

    # save winner name for report
    (OUT / "burst_winner.json").write_text(json.dumps({"winner": winner[0]}))


if __name__ == "__main__":
    main()
