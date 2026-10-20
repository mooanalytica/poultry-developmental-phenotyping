"""GRU sequence comparator with training-room-only imputation and sequences bounded to 24 real hours."""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
import models as M
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

SEQ_H = 24
MIN_OBS = 3
HP = dict(
    hidden=64,
    head=32,
    dropout=0.2,
    lr=1e-3,
    weight_decay=1e-4,
    loss="L1",
    epochs=40,
    batch=128,
    seed=0,
    layers=1,
)


def fit_imputer_scaler(train, cols):
    """Column medians, then mean/SD of the imputed training matrix. Uses ONLY the rows passed in (the training rooms)."""
    med = train[cols].median().fillna(0.0)
    filled = train[cols].fillna(med)
    mu = filled.mean()
    sd = filled.std().fillna(0.0) + 1e-6
    return med, mu, sd


def transform(df, cols, med, mu, sd):
    return ((df[cols].fillna(med) - mu) / sd).values.astype("float32")


def build_sequences(df, X):
    """df: one fold's evaluation windows (rows with room, hour, age_days), X: matching standardised features."""
    seqs, lens, ys, rooms, hours = [], [], [], [], []
    spans = []
    for room, g in df.groupby("room"):
        order = np.argsort(g.hour.values)
        idx = g.index.values[order]
        H = g.hour.values[order]
        A = g.age_days.values[order]
        V = X[df.index.get_indexer(idx)]
        lo = np.searchsorted(H, H - np.timedelta64(SEQ_H - 1, "h"), side="left")
        for i in range(len(H)):
            n = i - lo[i] + 1
            if n < MIN_OBS:
                continue
            dt = ((H[i] - H[lo[i] : i + 1]) / np.timedelta64(1, "h")).astype("float32")
            assert dt.max() <= SEQ_H - 1, "sequence spans more than 24 h of real time"
            seq = np.concatenate([V[lo[i] : i + 1], (dt / SEQ_H)[:, None]], 1)
            pad = np.zeros((SEQ_H, seq.shape[1]), dtype="float32")
            pad[:n] = seq
            seqs.append(pad)
            lens.append(n)
            ys.append(A[i])
            rooms.append(room)
            hours.append(H[i])
            spans.append(float(dt.max()))
    return (
        np.array(seqs),
        np.array(lens),
        np.array(ys, dtype="float32"),
        np.array(rooms),
        np.array(hours),
        np.array(spans),
    )


class GRUReg(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.g = nn.GRU(d, HP["hidden"], num_layers=HP["layers"], batch_first=True)
        self.h = nn.Sequential(
            nn.Linear(HP["hidden"], HP["head"]),
            nn.ReLU(),
            nn.Dropout(HP["dropout"]),
            nn.Linear(HP["head"], 1),
        )

    def forward(self, x, lens=None):
        if lens is None:
            o, _ = self.g(x)
            return self.h(o[:, -1]).squeeze(-1)
        pk = pack_padded_sequence(x, torch.as_tensor(lens), batch_first=True, enforce_sorted=False)
        _, hn = self.g(pk)
        return self.h(hn[-1]).squeeze(-1)


def train_predict(xtr, ltr, ytr, xte, lte):
    torch.manual_seed(HP["seed"])
    rng = np.random.RandomState(HP["seed"])
    net = GRUReg(xtr.shape[2])
    opt = torch.optim.Adam(net.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
    lossf = nn.L1Loss()
    xtr, ytr = torch.tensor(xtr), torch.tensor(ytr)
    n = len(xtr)
    idx = np.arange(n)
    for ep in range(HP["epochs"]):
        net.train()
        rng.shuffle(idx)
        for i in range(0, n, HP["batch"]):
            b = idx[i : i + HP["batch"]]
            opt.zero_grad()
            lossf(net(xtr[b], None if ltr is None else ltr[b]), ytr[b]).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        return net(torch.tensor(xte), lte).numpy()


def legacy_span_audit(Eseq, feat):
    """Spans of the Revision 3 sequences (24 consecutive observed rows, stride 3), before and after its 72 h filter."""
    out = []
    for room, g in Eseq.groupby("room"):
        H = g.sort_values("hour").hour.values
        for i in range(0, len(g) - 24, 3):
            out.append((room, float((H[i + 23] - H[i]) / np.timedelta64(1, "h"))))
    return pd.DataFrame(out, columns=["room", "span_h"])


def run_legacy_gru(Eseq_all, FC, room):
    """Revision 3 GRU on the same features (all-room median fill, 24 consecutive rows, <= 72 h span, stride 3) for a like-for-like reference."""
    E = Eseq_all.copy()
    E[FC] = E[FC].fillna(E[FC].median())
    xs, ys, rs = [], [], []
    for r, g in E.groupby("room"):
        g = g.sort_values("hour")
        V = g[FC].values.astype("float32")
        A = g.age_days.values.astype("float32")
        H = g.hour.values
        for i in range(0, len(g) - 24, 3):
            if int((H[i + 23] - H[i]) / np.timedelta64(1, "h")) > 72:
                continue
            xs.append(V[i : i + 24])
            ys.append(A[i + 23])
            rs.append(r)
    Xs, Ys, Rs = np.array(xs), np.array(ys, dtype="float32"), np.array(rs)
    tr, te = Rs != room, Rs == room
    mu = Xs[tr].reshape(-1, Xs.shape[2]).mean(0)
    sd = Xs[tr].reshape(-1, Xs.shape[2]).std(0) + 1e-6
    p = train_predict(
        ((Xs[tr] - mu) / sd).astype("float32"),
        None,
        Ys[tr],
        ((Xs[te] - mu) / sd).astype("float32"),
        None,
    )
    return M.metrics(Ys[te], p), int(len(Xs))


if __name__ == "__main__":
    import datasets as D
    import preprocessing as P
    from common import ROOMS

    out = Path(sys.argv[sys.argv.index("--out") + 1])
    out.mkdir(parents=True, exist_ok=True)
    rows, audit, sp_rows = [], [], []
    for room in ROOMS:
        d = P.fold("LORO", room)
        FC = d["fcols"]
        E = (
            d["Xw"][d["Xw"].window_h == D.WINDOWS_EVAL]
            .dropna(subset=["age_days"])
            .sort_values(["room", "hour"])
            .reset_index(drop=True)
        )
        trm = E.room != room
        med, mu, sd = fit_imputer_scaler(E[trm], FC)  # TRAINING ROOMS ONLY
        X = transform(E, FC, med, mu, sd)
        S, L, Y, R, H, SP = build_sequences(E, X)
        tr, te = R != room, R == room
        pred = train_predict(S[tr], L[tr], Y[tr], S[te], L[te])
        met = M.metrics(Y[te], pred)
        # HGB fused (primary) on the same anchors, and on all rows
        hg = P.fit_fold("LORO", room)["pred"].copy()
        hg["hour"] = pd.to_datetime(hg.hour)
        key = pd.DataFrame({"hour": pd.to_datetime(H[te]), "y": Y[te], "gru": pred})
        mg = key.merge(hg[["hour", "pred"]], on="hour", how="left")
        hs = (
            M.metrics(mg.y, mg.pred)
            if mg.pred.notna().all()
            else M.metrics(mg.dropna().y, mg.dropna().pred)
        )
        ha = P.fit_fold("LORO", room)["met"]
        lm, nseq = run_legacy_gru(E, FC, room)
        rows += [
            dict(
                heldout=room,
                model="GRU, time-bounded (this fix)",
                n_eval=met["n"],
                n_train_sequences=int(tr.sum()),
                MAE=met["MAE"],
                RMSE=met["RMSE"],
                R2=met["R2"],
                median_span_h=float(np.median(SP[te])),
                max_span_h=float(SP[te].max()),
                median_obs_per_sequence=float(np.median(L[te])),
            ),
            dict(
                heldout=room,
                model="HGB fused (primary), same anchors as the GRU",
                n_eval=int(len(mg)),
                MAE=hs["MAE"],
                RMSE=hs["RMSE"],
                R2=hs["R2"],
            ),
            dict(
                heldout=room,
                model="HGB fused (primary), all held-out windows",
                n_eval=ha["n"],
                MAE=ha["MAE"],
                RMSE=ha["RMSE"],
                R2=ha["R2"],
            ),
            dict(
                heldout=room,
                model="GRU, Revision 3 construction on the primary features",
                n_eval=lm["n"],
                n_train_sequences=nseq,
                MAE=lm["MAE"],
                RMSE=lm["RMSE"],
                R2=lm["R2"],
            ),
        ]
        sa = legacy_span_audit(E, FC)
        sa["heldout_fold"] = room
        sp_rows.append(sa)
    R_ = pd.DataFrame(rows)
    mac = R_.groupby("model")[["MAE", "RMSE", "R2"]].mean().reset_index().assign(heldout="macro")
    pd.concat([R_, mac], ignore_index=True).to_csv(out / "set3b_gru_vs_hgb.csv", index=False)
    SPN = pd.concat(sp_rows)
    aud = (
        SPN[SPN.heldout_fold == "room2"]
        .groupby("room")
        .span_h.agg(
            n_candidate_sequences="size",
            median_span_h="median",
            p90_span_h=lambda s: s.quantile(0.9),
            max_span_h="max",
            share_gt_24h=lambda s: (s > 24).mean(),
            share_gt_72h=lambda s: (s > 72).mean(),
        )
        .reset_index()
    )
    aud["share_kept_by_72h_filter"] = 1 - aud.share_gt_72h
    aud["share_of_kept_sequences_longer_than_24h"] = [
        float(((g.span_h > 24) & (g.span_h <= 72)).sum() / (g.span_h <= 72).sum())
        for _, g in SPN[SPN.heldout_fold == "room2"].groupby("room")
    ]
    aud.to_csv(out / "set3b_legacy_sequence_spans.csv", index=False)
    print(pd.concat([R_, mac], ignore_index=True).round(3).to_string(index=False))
    print(aud.round(3).to_string(index=False))
