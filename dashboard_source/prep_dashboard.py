"""用法：
  1. 從 https://github.com/tidyverse/nycflights13/tree/main/data 下載 flights.rda、weather.rda，放進 data/ 資料夾
  2. python prep_dashboard.py      → 產生 dash_data.json 與 index.html（需 pandas、numpy）

Prepare data for the interactive PCA dashboard (mirrors the user's R pipeline: strategies A–D, prcomp scale=TRUE,
signs aligned to the user's R output for strategy D)."""
import sys, json, numpy as np, pandas as pd
import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rda_reader import read_rda
D = os.path.join(HERE, "data") + os.sep
f = read_rda(D + "flights.rda")["flights"]; w = read_rda(D + "weather.rda")["weather"]
fd = f.merge(w, on=["origin", "time_hour"], how="left", suffixes=("_flight", "_weather"))
fv = ["dep_delay", "arr_delay", "air_time", "distance"]; wv = ["temp", "dewp", "humid", "wind_speed", "precip", "visib", "pressure"]
V = fv + wv
ad = fd[fd.dep_time.notna() & fd[fv].notna().all(axis=1)].reset_index(drop=True)
ad["p_missing"] = ad.pressure.isna()
A = ad[ad[V].notna().all(axis=1)]
B = ad.copy()
for v in wv: B[v] = B[v].fillna(ad[v].median())
gm = ad.groupby(["origin", "month_flight"]).pressure.transform("median")
C = B.copy(); C.loc[ad.p_missing, "pressure"] = gm[ad.p_missing]
grid = pd.date_range(w.time_hour.min(), w.time_hour.max(), freq="h", tz="America/New_York")
G = []
for o, g in w.groupby("origin"):
    g = g.set_index("time_hour").sort_index(); g = g[~g.index.duplicated()].reindex(grid); g["origin"] = o; G.append(g[["origin", "pressure"]])
G = pd.concat(G).rename_axis("time_hour").reset_index()
G["p"] = G.groupby("origin").pressure.transform(lambda s: s.interpolate(limit_area="inside"))
off = G.groupby("origin").pressure.mean() - G.pressure.mean()
G["p"] = G.p.fillna(G.groupby("time_hour").pressure.transform("mean") + G.origin.map(off))
G["p"] = G.groupby("origin").p.transform(lambda s: s.ffill().bfill())
Dd = C.drop(columns="pressure").merge(G[["origin", "time_hour", "p"]].rename(columns={"p": "pressure"}), on=["origin", "time_hour"], how="left")
Dd["pressure"] = Dd.pressure.fillna(pd.Series(C.pressure.values, index=Dd.index))
S = {"A": A, "B": B, "C": C, "D": Dd}

def pca(d):
    X = d[V].values.astype(float); mu, sd = X.mean(0), X.std(0, ddof=1); Z = (X - mu) / sd
    ev, vec = np.linalg.eigh(np.cov(Z, rowvar=False)); o = np.argsort(ev)[::-1]
    return {"ev": ev[o], "vec": vec[:, o], "mu": mu, "sd": sd}
P = {k: pca(d) for k, d in S.items()}
# sign convention = user's R output for D
rules = [("dewp", 1), ("distance", 1), ("dep_delay", -1), ("visib", -1), ("wind_speed", 1)]
for j, (var, sgn) in enumerate(rules):
    i = V.index(var)
    if np.sign(P["D"]["vec"][i, j]) != sgn: P["D"]["vec"][:, j] *= -1
for k in "ABC":
    for j in range(5):
        if P[k]["vec"][:, j] @ P["D"]["vec"][:, j] < 0: P[k]["vec"][:, j] *= -1
K = 5
def scores(k, d):
    Z = (d[V].values.astype(float) - P[k]["mu"]) / P[k]["sd"]; return Z @ P[k]["vec"][:, :K]
full = Dd.copy()
sc = scores("D", Dd)
for j in range(K): full[f"PC{j+1}"] = sc[:, j]

# ---- verification of findings on the FULL data (strategy D) ----
pr = lambda *a: print(*a)
pr("loadings D:\n", pd.DataFrame((P["D"]["vec"][:, :K] * np.sqrt(P["D"]["ev"][:K])).round(3), index=V, columns=[f"PC{j+1}" for j in range(K)]))
g = full.groupby("p_missing")[["PC1", "PC2", "PC3", "arr_delay", "humid", "visib"]].mean().round(2); pr("by p_missing\n", g)
pr("share missing with PC1>1:", round((full[full.p_missing].PC1 > 1).mean(), 3), " observed:", round((full[~full.p_missing].PC1 > 1).mean(), 3))
pr(full.groupby(["origin", "p_missing"])[["PC1", "arr_delay"]].mean().round(2))
pr("PC2 by origin", full.groupby("origin").PC2.mean().round(2).to_dict(), full.groupby("origin").distance.mean().round(0).to_dict())
lm = full[(full.origin == "LGA") & (full.month_flight == 5)]
pr("LGA May n", lm.p_missing.value_counts().to_dict()); pr(lm.groupby("p_missing")[["PC1", "arr_delay", "humid"]].mean().round(2)); pr(lm.groupby("p_missing").arr_delay.median())
pr("PC1 by month", full.groupby("month_flight").PC1.mean().round(2).to_dict())
pr("PC3 by month", full.groupby("month_flight").PC3.mean().round(2).to_dict())
hi = full[full.PC1 > 2]; pr("PC1>2 n", len(hi), "pmiss%", round(hi.p_missing.mean()*100, 1), "arr mean", round(hi.arr_delay.mean(), 1), "humid", round(hi.humid.mean(), 1), "visib", round(hi.visib.mean(), 2), "months", hi.month_flight.value_counts(normalize=True).round(3).head(4).to_dict())
pr("all: pmiss%", round(full.p_missing.mean()*100, 1), "arr mean", round(full.arr_delay.mean(), 1), "humid", round(full.humid.mean(), 1))
lo3 = full[full.PC3 < -2]; pr("PC3<-2 n", len(lo3), "arr", round(lo3.arr_delay.mean(), 1), "temp", round(lo3.temp.mean(), 1), "months", lo3.month_flight.value_counts(normalize=True).round(3).head(5).to_dict(), "pmiss", round(lo3.p_missing.mean()*100,1))
# corr delay vs temp overall and by season
pr("corr arr_delay~temp overall", round(full[["arr_delay", "temp"]].corr().iloc[0, 1], 3))
for nm, ms in {"summer": [6, 7, 8], "winter": [12, 1, 2]}.items():
    s = full[full.month_flight.isin(ms)]; pr(nm, round(s[["arr_delay", "temp"]].corr().iloc[0, 1], 3), round(s[["arr_delay", "humid"]].corr().iloc[0, 1], 3))

# ---- sample for the dashboard (compact, column arrays) ----
rng = np.random.default_rng(2013)
N = 20000
idx = np.sort(rng.choice(len(Dd), N, replace=False))
samp = Dd.iloc[idx]
raw = ad.iloc[idx]
def sc100(a):
    return [None if np.isnan(x) else int(round(x * 100)) for x in a]
cols = {
    "o": [ {"EWR":0,"JFK":1,"LGA":2}[x] for x in samp.origin ],
    "m": samp.month_flight.astype(int).tolist(), "d": samp.day_flight.astype(int).tolist(),
    "h": (samp.sched_dep_time // 100).astype(int).tolist(),
    "c": samp.carrier.tolist(), "dst": samp.dest.tolist(),
    "pm": raw.pressure.isna().astype(int).tolist(),
    "arr": samp.arr_delay.astype(int).tolist(), "dep": samp.dep_delay.astype(int).tolist(),
    "dist": samp.distance.astype(int).tolist(),
    "temp": [None if np.isnan(x) else round(x,1) for x in raw.temp],
    "hum": [None if np.isnan(x) else round(x,1) for x in raw.humid],
    "vis": [None if np.isnan(x) else round(x,2) for x in raw.visib],
    "pre": [None if np.isnan(x) else round(x,2) for x in raw.precip],
    "wind": [None if np.isnan(x) else round(x,1) for x in raw.wind_speed],
    "praw": [None if np.isnan(x) else round(x,1) for x in raw.pressure],
    "pD": samp.pressure.round(1).tolist(),
}
scoresS = {}
for k in S:
    if k == "A":
        X = raw[V].values.astype(float)
    else:
        X = S[k].iloc[idx][V].values.astype(float)
    Z = (X - P[k]["mu"]) / P[k]["sd"]; sc = Z @ P[k]["vec"][:, :K]
    scoresS[k] = [sc100(sc[:, j]) for j in range(K)]
strat = {k: {"ev": P[k]["ev"].round(4).tolist(), "load": (P[k]["vec"][:, :K] * np.sqrt(P[k]["ev"][:K])).round(3).tolist(), "n": int(len(S[k]))} for k in S}
# full-data cube by origin x month x pressure-missing, for exact group means in the dashboard
cube = []
pmf = ad.pressure.isna().values
for k in S:
    pass
fullsc = {}
for k in S:
    if k == "A":
        okm = ad[V].notna().all(axis=1).values
        X = ad[V].values.astype(float)
    else:
        okm = np.ones(len(ad), bool); X = S[k][V].values.astype(float)
    Z = (X - P[k]["mu"]) / P[k]["sd"]; scf = Z @ P[k]["vec"][:, :K]
    fullsc[k] = (okm, scf)
for oi, o in enumerate(["EWR", "JFK", "LGA"]):
    for mo in range(1, 13):
        for pmv in (0, 1):
            msk = (ad.origin.values == o) & (ad.month_flight.values == mo) & (pmf == pmv)
            cell = {"o": oi, "m": mo, "pm": pmv, "n": int(msk.sum()), "arr": float(ad.arr_delay.values[msk].sum()),
                    "hum": float(np.nansum(ad.humid.values[msk])), "humn": int(np.isfinite(ad.humid.values[msk]).sum())}
            for k in S:
                okm, scf = fullsc[k]; mm = msk & okm
                cell["n" + k] = int(mm.sum()); cell["s" + k] = [round(float(scf[mm, j].sum()), 2) for j in range(K)]
            cube.append(cell)
data = {"vars": V, "strat": strat, "pts": cols, "sc": scoresS, "N": N, "Nfull": int(len(Dd)), "cube": cube}
txt = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
open(os.path.join(HERE, "dash_data.json"), "w", encoding="utf-8").write(txt)
tpl = open(os.path.join(HERE, "page_template.html"), encoding="utf-8").read()
i = tpl.index('<div class="wrap">')
page = ('<!doctype html>\n<html lang="zh-Hant">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<style>:root{color-scheme:light}body{margin:0}[hidden]{display:none!important}</style>\n'
        + tpl[:i] + '</head>\n<body>\n' + tpl[i:].replace("__DATA__", txt) + '\n</body>\n</html>\n')
open(os.path.join(HERE, "index.html"), "w", encoding="utf-8").write(page)
print("json KB", len(txt)//1024)
# sample-level checks (what the dashboard will show)
sm = samp.assign(pm=raw.pressure.isna().values, PC1=scoresS["D"][0], PC2=scoresS["D"][1], PC3=scoresS["D"][2])
sm["PC1"] = sm.PC1/100; sm["PC2"]=sm.PC2/100; sm["PC3"]=sm.PC3/100
print("sample by pm", sm.groupby("pm")[["PC1","arr_delay"]].mean().round(2).to_dict())
l = sm[(sm.origin=="LGA")&(sm.month_flight==5)]
print("sample LGA May", l.pm.value_counts().to_dict(), l.groupby("pm").arr_delay.agg(["mean","median"]).round(1).to_dict(), l.groupby("pm").PC1.mean().round(2).to_dict())
h = sm[sm.PC1>2]; print("sample PC1>2", len(h), round(h.pm.mean()*100,1), round(h.arr_delay.mean(),1), round(raw.loc[h.index].humid.mean(),1))
l3 = sm[sm.PC3<-2]; print("sample PC3<-2", len(l3), round(l3.arr_delay.mean(),1), l3.month_flight.value_counts(normalize=True).round(3).head(5).to_dict())
