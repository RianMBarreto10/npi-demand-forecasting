"""
================================================================================
NPI DEMAND FORECASTING FRAMEWORK v5.0 — M5 Walmart Dataset
================================================================================
CHANGELOG v5.0 (adições sobre v4 — estrutura preservada):

  [IMP1] StockoutPreprocessor.flag_exogenous_anomalies()
         Detecta semanas com demanda ACIMA do P90 e cruza com proxies
         exógenas do M5 (snap_CA/TX/WI, eventos especiais no calendar).
         Retorna DataFrame com coluna 'exog_explanation' para cada pico.

  [IMP2] ForecastEvaluator: adiciona MASE e sMAPE como métricas
         complementares ao WAPE para semanas iniciais de baixo volume.
         MASE usa naive seasonal (lag-4 semanas) como benchmark.
         sMAPE é simétrico e robusto a zeros.

  [IMP3] DemandForecastEngine.fit_rampup_decay() ampliado:
         Agora testa TRÊS curvas em paralelo — Sigmoide, Gompertz e
         Richards (generalizada) — e seleciona a de menor RMSE via
         model_selection_report(). O parâmetro 'curve_source' indica
         qual curva venceu.

  [IMP4] CurveSelector (nova classe auxiliar):
         Encapsula o ajuste e comparação das 3 curvas com AIC/BIC/RMSE.
         Evita duplicação de código e facilita extensão futura.

  [IMP5] plot_all() atualizado:
         - Subplot 4 exibe WAPE + MASE + sMAPE (cold-start cinza)
         - Anotação no subplot 1 indica qual curva venceu + delta
         - Subplot 2 destaca picos exógenos com marcador especial (★)

CHANGELOG v4 (mantidos):
  [AJ1] WAPE bifásico cold-start vs steady-state (sem. MIN_EVAL_WEEK)
  [AJ2] Sigmoide × decaimento exponencial pós-pico (parâmetro δ)
================================================================================
"""

import os, sys
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize_scalar
from scipy.special import lambertw
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ==============================================================================
# CONFIGURAÇÃO
# ==============================================================================
M5_DATA_DIR           = "."
N_NPI_WEEKS           = 28
MIN_POSITIVE_WEEKS    = 8
MAX_POSITIVE_WEEKS    = 60
MIN_TOTAL_SALES       = 100
MIN_RAMPUP_SCORE      = 0.3
MIN_POSITIVE_FOR_FIT  = 6
N_SIMILAR             = 5
MC_SIMS               = 10_000
MIN_EVAL_WEEK         = 4      # [AJ1] separação cold-start / steady-state
P90_ANOMALY_FACTOR    = 1.5    # [IMP1] multiplier acima do P90 para flagrar pico


# ==============================================================================
# MÓDULO 0: CARREGAMENTO E PREPARAÇÃO
# ==============================================================================

def load_m5(data_dir):
    print("Carregando arquivos M5...")
    sales  = pd.read_csv(os.path.join(data_dir, "sales_train_evaluation.csv"))
    cal    = pd.read_csv(os.path.join(data_dir, "calendar.csv"))
    prices = pd.read_csv(os.path.join(data_dir, "sell_prices.csv"))
    print(f"  sales  : {sales.shape}  ({sales['item_id'].nunique()} items)")
    print(f"  calendar: {cal.shape}")
    print(f"  prices  : {prices.shape}")
    return sales, cal, prices


def melt_to_weekly(sales, cal):
    id_cols  = ["id","item_id","dept_id","cat_id","store_id","state_id"]
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    print(f"  Melt de {len(day_cols)} colunas diarias...")
    df_long = sales.melt(id_vars=id_cols, value_vars=day_cols,
                         var_name="d", value_name="sales")
    cal_map = cal[["d","wm_yr_wk","snap_CA","snap_TX","snap_WI",
                   "event_name_1","event_type_1",
                   "event_name_2","event_type_2"]].copy()
    df_long = df_long.merge(cal_map, on="d", how="left")
    df_weekly = (
        df_long
        .groupby(id_cols + ["wm_yr_wk"])
        .agg(
            weekly_sales=("sales","sum"),
            snap_CA=("snap_CA","max"),
            snap_TX=("snap_TX","max"),
            snap_WI=("snap_WI","max"),
            has_event=("event_name_1", lambda x: int(x.notna().any())),
        )
        .reset_index()
        .sort_values(["id","wm_yr_wk"])
    )
    print(f"  df_weekly: {df_weekly.shape}")
    return df_weekly


def score_rampup_quality(sales_series):
    s = np.array(sales_series, dtype=float)
    n = len(s)
    if s.sum() == 0 or n < 4:
        return 0.0
    mid = max(n // 2, 2)
    first_half = s[:mid]
    t = np.arange(mid)
    trend_score = max(np.corrcoef(t, first_half)[0,1], 0) if first_half.std() > 0 else 0.0
    second_half = s[mid:]
    spread_score = (second_half > 0).mean() if len(second_half) > 0 else 0.0
    pos_idx = np.where(s > 0)[0]
    if len(pos_idx) < 2:
        return 0.0
    span = pos_idx[-1] - pos_idx[0] + 1
    zeros_inside = (s[pos_idx[0]:pos_idx[-1]+1] == 0).sum()
    continuity_score = 1 - zeros_inside / span if span > 1 else 0.0
    return float(0.4*trend_score + 0.35*spread_score + 0.25*continuity_score)


def identify_npi_skus(df_weekly, min_pos=MIN_POSITIVE_WEEKS,
                      max_pos=MAX_POSITIVE_WEEKS, min_sales=MIN_TOTAL_SALES,
                      min_score=MIN_RAMPUP_SCORE, top_n=20):
    stats = (
        df_weekly.groupby("id")
        .agg(weeks_pos=("weekly_sales", lambda x: (x>0).sum()),
             total_sales=("weekly_sales","sum"))
        .reset_index()
    )
    candidates = stats[
        (stats["weeks_pos"] >= min_pos) &
        (stats["weeks_pos"] <= max_pos) &
        (stats["total_sales"] >= min_sales)
    ]["id"].tolist()
    print(f"  Pre-filtro: {len(candidates)} candidatos")
    scores = []
    for sku_id in candidates:
        s = df_weekly[df_weekly["id"]==sku_id]["weekly_sales"].values
        first = np.where(s > 0)[0]
        if len(first) == 0:
            continue
        s_ramp = s[first[0]:first[0]+N_NPI_WEEKS]
        score  = score_rampup_quality(s_ramp)
        if score >= min_score:
            scores.append({"id":sku_id, "score":score,
                           "total_sales": stats[stats["id"]==sku_id]["total_sales"].values[0]})
    df_scores = pd.DataFrame(scores).sort_values(["score","total_sales"], ascending=False)
    print(f"  Pos score_rampup >= {min_score}: {len(df_scores)} SKUs validos")
    if df_scores.empty:
        print("  Relaxando para min_score=0.1...")
        return identify_npi_skus(df_weekly, min_pos, max_pos, min_sales, 0.1, top_n)
    return df_scores["id"].head(top_n).tolist()


def extract_sku_rampup(df_weekly, sku_id, n_weeks=N_NPI_WEEKS):
    cols_base = ["id","item_id","dept_id","cat_id","store_id","state_id","wm_yr_wk","weekly_sales"]
    # agrega colunas exógenas apenas se existirem
    exog_cols = [c for c in ["snap_CA","snap_TX","snap_WI","has_event"] if c in df_weekly.columns]
    series = (
        df_weekly[df_weekly["id"]==sku_id][cols_base + exog_cols]
        .sort_values("wm_yr_wk").reset_index(drop=True)
    )
    first = series[series["weekly_sales"]>0].index.min()
    if pd.isna(first): first = 0
    series = series.iloc[first:].reset_index(drop=True)
    series["week_num"] = np.arange(1, len(series)+1)
    return series.head(n_weeks)


def diagnose_sku(sku_series, sku_id):
    s = sku_series["weekly_sales"].values
    score = score_rampup_quality(s)
    print(f"  ID           : {sku_id}")
    print(f"  Semanas      : {len(sku_series)}")
    print(f"  Com vendas   : {(s>0).sum()} semanas")
    print(f"  Zeros        : {(s==0).sum()} semanas")
    print(f"  Total        : {s.sum():,.0f} unidades")
    print(f"  Pico         : {s.max():,.0f} unid. (semana {s.argmax()+1})")
    print(f"  Rampup Score : {score:.3f}")


# ==============================================================================
# MÓDULO 1: PREPROCESSING + DETECÇÃO EXÓGENA [IMP1]
# ==============================================================================

class StockoutPreprocessor:
    """
    [IMP1] flag_exogenous_anomalies():
    Identifica semanas onde a demanda observada supera P90 × P90_ANOMALY_FACTOR
    e cruza com variáveis exógenas do M5 (SNAP benefits, eventos especiais).
    Retorna DataFrame com coluna 'exog_explanation' descrevendo cada pico.

    Lógica:
      - threshold = np.percentile(demand_positives, 90) × P90_ANOMALY_FACTOR
      - Se semana_i > threshold → pico anômalo
      - Para cada pico: verifica snap_CA/TX/WI e has_event da mesma semana
      - exog_explanation = lista de causas encontradas ou 'sem proxy exógeno'
    """

    def __init__(self, run_rate_window=3, global_impute_threshold=0.5):
        self.window    = run_rate_window
        self.threshold = global_impute_threshold

    def fit_transform(self, df):
        df     = df.copy().sort_values("week_num").reset_index(drop=True)
        df["stockout_flag"] = 0
        df["latent_demand"] = df["weekly_sales"].astype(float)
        sales  = df["weekly_sales"].values

        # Nível 1: zeros locais (ruptura)
        for i in range(self.window, len(df)):
            if sales[i] > 0: continue
            prior_pos  = any(sales[max(0,i-self.window):i] > 0)
            future_pos = any(sales[i+1:min(len(sales),i+6)] > 0)
            if prior_pos and future_pos:
                df.loc[i, "stockout_flag"] = 1
                rr = np.mean(sales[max(0,i-self.window):i][sales[max(0,i-self.window):i]>0])
                if np.isnan(rr) or rr == 0: rr = 1.0
                df.loc[i, "latent_demand"] = rr * np.random.uniform(0.92, 1.08)

        # Nível 2: ruptura global pós-pico
        pos_weeks = np.where(sales > 0)[0]
        if len(pos_weeks) >= 3:
            peak_idx = int(np.argmax(sales))
            post_peak = sales[peak_idx+1:]
            if len(post_peak) > 0 and (post_peak == 0).mean() >= self.threshold:
                pre_pos   = sales[:peak_idx+1][sales[:peak_idx+1] > 0]
                run_rate  = float(pre_pos.mean()) if len(pre_pos) > 0 else float(sales.max())
                for i in range(peak_idx+1, len(df)):
                    if df.loc[i,"stockout_flag"] == 0 and sales[i] == 0:
                        df.loc[i,"stockout_flag"] = 1
                        decay = np.exp(-0.05 * (i - peak_idx))
                        df.loc[i,"latent_demand"] = max(run_rate*decay*np.random.uniform(0.88,1.05), 0)
        return df

    @staticmethod
    def flag_exogenous_anomalies(sku_clean, p90_factor=P90_ANOMALY_FACTOR):
        """
        [IMP1] Detecta picos acima do P90 e cruza com proxies exógenas do M5.
        Retorna sku_clean com coluna adicional 'exog_explanation'.
        """
        df      = sku_clean.copy()
        demand  = df["latent_demand"].values
        pos_dem = demand[demand > 0]
        if len(pos_dem) < 4:
            df["exog_flag"]        = 0
            df["exog_explanation"] = ""
            return df

        p90_thresh = np.percentile(pos_dem, 90) * p90_factor
        exog_flags  = []
        exog_labels = []

        for _, row in df.iterrows():
            d = row["latent_demand"]
            if d <= p90_thresh:
                exog_flags.append(0)
                exog_labels.append("")
                continue

            # Pico detectado — verifica causas exógenas
            causes = []
            if "snap_CA" in df.columns and row.get("snap_CA", 0) == 1:
                causes.append("SNAP-CA")
            if "snap_TX" in df.columns and row.get("snap_TX", 0) == 1:
                causes.append("SNAP-TX")
            if "snap_WI" in df.columns and row.get("snap_WI", 0) == 1:
                causes.append("SNAP-WI")
            if "has_event" in df.columns and row.get("has_event", 0) == 1:
                causes.append("Evento")

            exog_flags.append(1)
            exog_labels.append(", ".join(causes) if causes else "sem proxy exogeno")

        df["exog_flag"]        = exog_flags
        df["exog_explanation"] = exog_labels
        return df


# ==============================================================================
# MÓDULO 2: COLD START
# ==============================================================================

class M5SimilarityEngine:
    def __init__(self, n_clusters=8, n_similar=N_SIMILAR):
        self.n_clusters  = n_clusters
        self.n_similar   = n_similar
        self.scaler      = StandardScaler()
        self.kmeans      = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.features_df = None
        self.rampup_db   = {}

    def build_features(self, df_weekly, prices):
        price_stats = (
            prices.groupby(["item_id","store_id"])["sell_price"]
            .agg(price_mean="mean", price_std="std").reset_index()
        )
        price_stats["price_cv"] = (
            price_stats["price_std"] / price_stats["price_mean"].replace(0,np.nan)
        ).fillna(0)
        vol = (
            df_weekly[df_weekly["weekly_sales"]>0]
            .groupby("id")["weekly_sales"].mean().reset_index()
            .rename(columns={"weekly_sales":"avg_weekly"})
        )
        meta = df_weekly[["id","item_id","dept_id","cat_id","store_id","state_id"]].drop_duplicates("id")
        meta = meta.merge(price_stats, on=["item_id","store_id"], how="left")
        meta = meta.merge(vol, on="id", how="left")
        for col in ["dept_id","cat_id","store_id","state_id"]:
            meta[col+"_enc"] = pd.factorize(meta[col])[0].astype(float)
        fcols = ["dept_id_enc","cat_id_enc","store_id_enc","state_id_enc",
                 "price_mean","price_cv","avg_weekly"]
        meta[fcols] = meta[fcols].fillna(0)
        self.features_df = meta
        return meta

    def fit(self, rampup_db):
        self.rampup_db = rampup_db
        fcols = ["dept_id_enc","cat_id_enc","store_id_enc","state_id_enc",
                 "price_mean","price_cv","avg_weekly"]
        X = self.scaler.fit_transform(self.features_df[fcols].fillna(0))
        self.kmeans.fit(X)
        self.features_df = self.features_df.copy()
        self.features_df["cluster"] = self.kmeans.labels_
        return self

    def find_similar(self, target_id):
        fcols = ["dept_id_enc","cat_id_enc","store_id_enc","state_id_enc",
                 "price_mean","price_cv","avg_weekly"]
        if target_id not in self.features_df["id"].values:
            return {"similar_ids":[], "inherited_curve":None, "top_similarity":0}
        tgt_feat = self.scaler.transform(
            self.features_df[self.features_df["id"]==target_id][fcols].fillna(0)
        )
        all_feat = self.scaler.transform(self.features_df[fcols].fillna(0))
        sims     = cosine_similarity(tgt_feat, all_feat)[0]
        self.features_df = self.features_df.copy()
        self.features_df["similarity"] = sims
        top     = self.features_df[self.features_df["id"]!=target_id].nlargest(self.n_similar,"similarity")
        curves  = [self.rampup_db[sid] for sid in top["id"].values if sid in self.rampup_db]
        inherited = None
        if curves:
            ml     = max(len(c) for c in curves)
            padded = [np.pad(c,(0,ml-len(c)),mode="edge") for c in curves]
            inherited = np.mean(padded, axis=0)
        return {
            "similar_ids":     top["id"].tolist(),
            "top_similarity":  float(top["similarity"].max()),
            "target_cluster":  int(self.features_df[self.features_df["id"]==target_id]["cluster"].values[0]),
            "inherited_curve": inherited,
        }


# ==============================================================================
# MÓDULO 3A: SELEÇÃO DE CURVAS [IMP3] [IMP4]
# ==============================================================================

class CurveSelector:
    """
    [IMP3] [IMP4] Testa Sigmoide, Gompertz e Richards em paralelo.
    Seleciona a curva com menor RMSE e calcula AIC/BIC para cada modelo.

    Curvas:
      Sigmoid  : L / (1 + exp(-k*(t-t0)))
      Gompertz : L * exp(-exp(-k*(t-t0)))
      Richards : L / (1 + v*exp(-k*(t-t0)))^(1/v)   [generalização de ambas]

    Por que Gompertz?
      A sigmoide tem ponto de inflexão fixo em L/2.
      A Gompertz tem inflexão em L/e (~37% do pico) — modela adoção
      assimétrica onde o crescimento inicial é mais lento que o declínio.
      Isso é mais realista para categorias de nicho/luxo.

    Por que Richards?
      Generalização com parâmetro de forma v — recupera Sigmoid (v→1)
      e Gompertz (v→0) como casos especiais. Maior flexibilidade,
      mas penalizado por AIC/BIC por ter 1 parâmetro a mais.
    """

    MODELS = ["sigmoid", "gompertz", "richards"]

    @staticmethod
    def sigmoid(t, L, k, t0):
        return L / (1.0 + np.exp(-k * (t - t0)))

    @staticmethod
    def gompertz(t, L, k, t0):
        return L * np.exp(-np.exp(-k * (t - t0)))

    @staticmethod
    def richards(t, L, k, t0, v):
        v  = max(v, 0.01)
        ex = np.exp(-k * (t - t0))
        return L / (1.0 + v * ex) ** (1.0 / v)

    def fit_all(self, weeks, demand, inherited_curve=None):
        """
        Ajusta as 3 curvas e retorna dict com parâmetros + métricas.
        Se o ajuste direto falhar, tenta com a curva herdada escalada.
        """
        t = weeks.astype(float)
        d = demand.astype(float)
        L_max   = d.max() * 1.5 if d.max() > 0 else 10.0
        grad    = np.gradient(d)
        t0_init = float(t[np.argmax(grad)]) if grad.max() > 0 else float(t[len(t)//3])
        results = {}

        # ── Sigmoid ────────────────────────────────────────────────────────────
        try:
            popt, _ = curve_fit(
                self.sigmoid, t, d, p0=[L_max, 0.25, t0_init],
                bounds=([d.max()*0.4, 0.01, 1.0], [d.max()*6.0, 3.0, float(t.max())]),
                maxfev=15000
            )
            pred = self.sigmoid(t, *popt)
            results["sigmoid"] = self._pack("sigmoid", popt, t, d, pred, n_params=3)
        except Exception:
            results["sigmoid"] = None

        # ── Gompertz ───────────────────────────────────────────────────────────
        try:
            popt, _ = curve_fit(
                self.gompertz, t, d, p0=[L_max, 0.25, t0_init],
                bounds=([d.max()*0.4, 0.01, -5.0], [d.max()*6.0, 3.0, float(t.max())+5]),
                maxfev=15000
            )
            pred = self.gompertz(t, *popt)
            results["gompertz"] = self._pack("gompertz", popt, t, d, pred, n_params=3)
        except Exception:
            results["gompertz"] = None

        # ── Richards ───────────────────────────────────────────────────────────
        try:
            popt, _ = curve_fit(
                self.richards, t, d, p0=[L_max, 0.25, t0_init, 1.0],
                bounds=([d.max()*0.4, 0.01, 1.0, 0.01],
                        [d.max()*6.0,  3.0, float(t.max()), 10.0]),
                maxfev=20000
            )
            pred = self.richards(t, *popt)
            results["richards"] = self._pack("richards", popt, t, d, pred, n_params=4)
        except Exception:
            results["richards"] = None

        # Fallback: se todos falharam, usa inherited_curve ou heurística
        valid = {k: v for k, v in results.items() if v is not None}
        if not valid:
            return self._fallback(t, d, inherited_curve)

        # Seleciona menor RMSE
        best_name = min(valid, key=lambda k: valid[k]["rmse"])
        return valid[best_name], results

    @staticmethod
    def _pack(name, popt, t, d, pred, n_params):
        resid = d - pred
        sse   = np.sum(resid**2)
        n     = len(d)
        rmse  = float(np.sqrt(sse / n))
        aic   = float(n * np.log(sse/n + 1e-12) + 2 * n_params)
        bic   = float(n * np.log(sse/n + 1e-12) + n_params * np.log(n))
        return {"name": name, "params": popt, "rmse": rmse,
                "aic": aic, "bic": bic, "pred": pred}

    def _fallback(self, t, d, inherited_curve):
        L_max = float(d.max() * 1.5) if d.max() > 0 else 10.0
        if inherited_curve is not None and len(inherited_curve) >= 4:
            inh = np.array(inherited_curve[:len(t)], dtype=float)
            if inh.max() > 0 and d.max() > 0:
                scale = (d[d>0].mean() / max(inh[inh>0].mean(), 1e-9))
                inh   = inh * scale
                try:
                    popt, _ = curve_fit(
                        self.sigmoid, t, inh,
                        p0=[inh.max()*1.3, 0.25, float(t[inh.argmax()])],
                        bounds=([inh.max()*0.3, 0.01, 1.0],
                                [inh.max()*5.0,  3.0, float(t.max())]),
                        maxfev=15000
                    )
                    pred = self.sigmoid(t, *popt)
                    res  = self._pack("sigmoid_herdado", popt, t, inh, pred, 3)
                    return res, {"sigmoid_herdado": res}
                except Exception:
                    pass
        # Heurística mínima
        res = {"name":"heuristica", "params":[L_max, 0.25, float(t[len(t)//3])],
               "rmse": float(d.std()+1), "aic":9999, "bic":9999,
               "pred": self.sigmoid(t, L_max, 0.25, float(t[len(t)//3]))}
        return res, {"heuristica": res}

    def model_selection_report(self, all_results):
        """Imprime tabela comparativa AIC/BIC/RMSE."""
        rows = [(v["name"], v["rmse"], v["aic"], v["bic"])
                for v in all_results.values() if v is not None]
        if not rows:
            return
        print("  ┌─────────────────┬──────────┬──────────┬──────────┐")
        print("  │ Curva           │   RMSE   │    AIC   │    BIC   │")
        print("  ├─────────────────┼──────────┼──────────┼──────────┤")
        for name, rmse, aic, bic in sorted(rows, key=lambda x: x[1]):
            marker = " ★" if rows.index((name,rmse,aic,bic)) == 0 or                      rmse == min(r[1] for r in rows) else "  "
            print(f"  │ {name:<15s} │ {rmse:8.2f} │ {aic:8.1f} │ {bic:8.1f} │{marker}")
        print("  └─────────────────┴──────────┴──────────┴──────────┘")


# ==============================================================================
# MÓDULO 3B: MOTOR DE PREVISÃO — [AJ2] + [IMP3]
# ==============================================================================

class DemandForecastEngine:
    """
    v5: integra CurveSelector para ajuste automático da melhor curva.
    Mantém o decaimento exponencial pós-pico do v4 [AJ2].
    """

    def __init__(self, market_potential=None):
        self.M              = market_potential
        self.sigmoid_params = {}
        self.decay_delta    = 0.0
        self.t_peak         = None
        self.bayesian_state = {}
        self.curve_selector = CurveSelector()
        self._best_curve    = None

    def _apply_decay(self, t, base_pred, t_peak, delta):
        decay = np.exp(-delta * np.maximum(0.0, t - t_peak))
        return base_pred * decay

    def _estimate_delta(self, weeks, demand, base_pred, t_peak):
        post = weeks > t_peak
        if post.sum() < 3:
            return 0.0
        def mse_d(delta):
            pred = self._apply_decay(weeks, base_pred, t_peak, delta)
            return np.mean((pred[post] - demand[post])**2)
        res = minimize_scalar(mse_d, bounds=(0.0, 0.30), method="bounded")
        return float(res.x)

    def fit_rampup_decay(self, weeks, demand, inherited_curve=None):
        t = weeks.astype(float)
        d = demand.astype(float)

        best, all_res = self.curve_selector.fit_all(t, d, inherited_curve)
        self.curve_selector.model_selection_report(all_res)
        self._best_curve = best

        # Extrai parâmetros da curva vencedora para uso comum
        popt   = best["params"]
        name   = best["name"]
        base   = best["pred"]
        t_peak = float(t[base.argmax()]) if base.max() > 0 else float(t[len(t)//3])

        # Estima decaimento sobre a curva vencedora
        delta = self._estimate_delta(t, d, base, t_peak)

        self.sigmoid_params = {
            "name":   name,
            "params": popt,
            "L":      float(popt[0]),
            "k":      float(popt[1]) if len(popt) > 1 else 0.25,
            "t0":     float(popt[2]) if len(popt) > 2 else t_peak,
            "source": name.upper(),
        }
        self.decay_delta = delta
        self.t_peak      = t_peak
        print(f"  -> Curva selecionada: {name.upper()} | RMSE={best['rmse']:.2f}")
        print(f"  -> Decaimento delta={delta:.4f} ({'ativo' if delta>0.01 else 'negligivel'})")
        return self.sigmoid_params

    # Alias de compatibilidade
    def fit_sigmoid(self, weeks, demand, inherited_curve=None):
        return self.fit_rampup_decay(weeks, demand, inherited_curve)

    def _predict_curve(self, t):
        if self._best_curve is None:
            return np.zeros_like(t)
        name   = self._best_curve["name"]
        popt   = self._best_curve["params"]
        if "sigmoid" in name:
            return CurveSelector.sigmoid(t, *popt)
        elif name == "gompertz":
            return CurveSelector.gompertz(t, *popt)
        elif name == "richards":
            return CurveSelector.richards(t, *popt)
        else:
            return CurveSelector.sigmoid(t, *popt[:3])

    def bass_diffusion(self, n_weeks, p=0.03, q=0.38):
        M = self.M or 10000
        N, n_, innov, imit = (np.zeros(n_weeks) for _ in range(4))
        N[0] = M*p; n_[0] = innov[0] = N[0]
        for t in range(1, n_weeks):
            innov[t] = p * (M - N[t-1])
            imit[t]  = q * N[t-1] / M * (M - N[t-1])
            n_[t]    = innov[t] + imit[t]
            N[t]     = N[t-1] + n_[t]
        return {"total":n_, "innovators":innov, "imitators":imit, "cumulative":N}

    def bayesian_update(self, obs_sales, prior_mu=1.0,
                        prior_sigma=0.20, obs_noise=0.12):
        valid = obs_sales[obs_sales > 0]
        if len(valid) == 0:
            self.bayesian_state = {"mu":1.0, "sigma":prior_sigma}
            return self.bayesian_state
        n       = len(valid)
        pos_idx = np.where(obs_sales > 0)[0]
        exp     = self._predict_curve((pos_idx+1).astype(float))
        exp     = np.maximum(exp, 1e-9)
        ratio   = float(np.mean(valid / exp))
        s2_pr   = prior_sigma**2
        s2_ob   = obs_noise**2
        s2_po   = 1.0 / (1.0/s2_pr + n/s2_ob)
        mu_po   = s2_po * (prior_mu/s2_pr + n*ratio/s2_ob)
        mu_po   = float(np.clip(mu_po, 0.5, 2.0))
        self.bayesian_state = {"mu":mu_po, "sigma":float(np.sqrt(s2_po))}
        return self.bayesian_state

    def predict(self, n_weeks, media_lift=None):
        t    = np.arange(1, n_weeks+1, dtype=float)
        base = self._predict_curve(t)
        base = self._apply_decay(t, base, self.t_peak or float(n_weeks/2), self.decay_delta)
        if self.bayesian_state.get("mu", 1.0) != 1.0:
            base = base * self.bayesian_state["mu"]
        if media_lift is not None:
            base = base * media_lift
        return pd.DataFrame({"week": np.arange(1, n_weeks+1), "p50": base})


# ==============================================================================
# MÓDULO 4: MONTE CARLO
# ==============================================================================

class MonteCarloProbabilisticForecast:
    def __init__(self, n_simulations=MC_SIMS, seed=42):
        self.n_sim = n_simulations
        np.random.seed(seed)

    def simulate(self, base_forecast, demand_cv=0.12):
        n    = len(base_forecast)
        sims = np.zeros((self.n_sim, n))
        for i in range(self.n_sim):
            noise = np.random.normal(1.0, demand_cv, n)
            trend = np.random.normal(0, 0.05)
            shock = np.random.choice([0.75,1.0,1.25], p=[0.08,0.85,0.07])
            sims[i] = np.maximum(base_forecast * noise * (1+trend) * shock, 0)
        return {
            "p10": np.percentile(sims, 10, axis=0),
            "p50": np.percentile(sims, 50, axis=0),
            "p90": np.percentile(sims, 90, axis=0),
        }


# ==============================================================================
# MÓDULO 5: AVALIAÇÃO + MÉTRICAS ADICIONAIS [IMP2] + PLOTS
# ==============================================================================

class ForecastEvaluator:
    """
    [AJ1] WAPE bifásico cold-start / steady-state
    [IMP2] MASE e sMAPE adicionados como métricas complementares.

    MASE (Mean Absolute Scaled Error):
        MASE = MAE_model / MAE_naive
        naive = forecast ingênuo lag-4 (último mês)
        MASE < 1 → modelo supera naive. MASE ≥ 1 → revisão necessária.
        Robusto a volumes baixos porque não divide por valores próximos de zero.

    sMAPE (Symmetric MAPE):
        sMAPE = 200 * |A-F| / (|A| + |F| + epsilon)
        Simétrico e limitado a [0, 200%].
        Mais estável que MAPE quando A → 0, e reportado no M5 Competition.
    """

    def __init__(self, min_eval_week=MIN_EVAL_WEEK):
        self.min_eval_week = min_eval_week

    @staticmethod
    def _wape(actual, forecast):
        d = np.sum(np.abs(actual))
        return float(np.sum(np.abs(actual - forecast)) / d * 100) if d > 0 else np.nan

    @staticmethod
    def _bias(actual, forecast):
        d = np.sum(np.abs(actual))
        return float(np.sum(forecast - actual) / d * 100) if d > 0 else np.nan

    @staticmethod
    def _mase(actual, forecast, lag=4):
        """
        [IMP2] MASE com naive lag-4.
        Benchmark: MAE de prever que semana t = semana t-4.
        """
        n = len(actual)
        if n <= lag:
            return np.nan
        naive_err = np.abs(actual[lag:] - actual[:-lag])
        mae_naive = float(naive_err.mean())
        if mae_naive < 1e-9:
            return np.nan
        mae_model = float(np.mean(np.abs(actual - forecast)))
        return mae_model / mae_naive

    @staticmethod
    def _smape(actual, forecast, eps=1.0):
        """
        [IMP2] sMAPE simétrico. eps evita divisão por zero.
        Resultado em [0, 200].
        """
        num   = np.abs(actual - forecast)
        denom = np.abs(actual) + np.abs(forecast) + eps
        return float(200.0 * np.mean(num / denom))

    @staticmethod
    def tracking_signal(actual, forecast):
        err   = forecast - actual
        cusum = np.cumsum(err)
        mad   = np.array([np.mean(np.abs(err[:i+1])) for i in range(len(err))])
        mad[mad == 0] = 1e-9
        return cusum / mad

    def wape_by_phase(self, actual, p50, weeks):
        cold_mask   = weeks < self.min_eval_week
        steady_mask = weeks >= self.min_eval_week
        return {
            "WAPE_coldstart_%":   round(self._wape(actual[cold_mask], p50[cold_mask]), 1)
                                  if cold_mask.sum() > 0 else "N/A",
            "WAPE_steadystate_%": round(self._wape(actual[steady_mask], p50[steady_mask]), 1)
                                  if steady_mask.sum() > 0 else "N/A",
        }

    def evaluate(self, actual, p10, p50, p90, weeks=None):
        if weeks is not None:
            mask = weeks >= self.min_eval_week
        else:
            mask = np.ones(len(actual), dtype=bool)

        act_ev = actual[mask]; p50_ev = p50[mask]
        p10_ev = p10[mask];    p90_ev = p90[mask]

        wv  = self._wape(act_ev, p50_ev)
        bv  = self._bias(act_ev, p50_ev)
        mv  = self._mase(act_ev, p50_ev)
        sv  = self._smape(act_ev, p50_ev)
        ts  = self.tracking_signal(act_ev, p50_ev)
        ic  = float(np.mean((act_ev >= p10_ev) & (act_ev <= p90_ev)) * 100)
        st  = ("EXCELENTE" if not np.isnan(wv) and wv < 10
               else "BOM"       if not np.isnan(wv) and wv < 20
               else "ACEITAVEL" if not np.isnan(wv) and wv < 30
               else "REVISAR")
        return {
            "WAPE_%":           round(wv,2)  if not np.isnan(wv) else "N/A",
            "BIAS_%":           round(bv,2)  if not np.isnan(bv) else "N/A",
            "MASE":             round(mv,3)  if not np.isnan(mv) else "N/A",
            "sMAPE_%":          round(sv,1)  if not np.isnan(sv) else "N/A",
            "IC_Coverage_%":    round(ic,1),
            "TS_Max":           round(float(abs(ts).max()),2),
            "Status":           st,
            "Semanas_avaliadas":int(mask.sum()),
        }

    # ──────────────────────────────────────────────────────────────────────────
    def plot_all(self, weeks, actual, p10, p50, p90,
                 innov, imit, stockout_flags,
                 exog_flags=None, exog_labels=None,
                 item_id="SKU", sigmoid_source="", decay_delta=0.0):

        fig = plt.figure(figsize=(18, 11))
        fig.patch.set_facecolor("#0d0d0f")
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.44, wspace=0.28,
                               left=0.07, right=0.97, top=0.93, bottom=0.08)
        fig.text(0.5, 0.98,
                 f"NPI Demand Forecasting Framework v5.0 — {item_id}",
                 ha="center", va="top", color="#e8e8ec",
                 fontsize=13, fontweight="bold")

        DARK="#131316"; GRID_C="#1f1f24"
        TEAL="#4f98a3"; GREEN="#6daa45"; RED="#dd6974"
        GOLD="#e8af34"; PURPLE="#a86fdf"; MUTED="#8888a0"; TEXT="#e8e8ec"

        def style_ax(ax, title, xlabel, ylabel):
            ax.set_facecolor(DARK)
            for sp in ax.spines.values(): sp.set_color("#2a2a30")
            ax.tick_params(colors=MUTED, labelsize=9)
            ax.set_title(title, color=TEXT, fontsize=10, fontweight="bold", pad=8)
            ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
            ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
            ax.grid(axis="y", color=GRID_C, lw=0.6, alpha=0.6)
            ax.set_xlim(0.0, len(weeks)+1)

        fmt_u = mticker.FuncFormatter(lambda x, _: f"{x:.0f}")

        # ── 1. RAMP-UP ────────────────────────────────────────────────────────
        ax1 = fig.add_subplot(gs[0,0])
        decay_str = f" | delta={decay_delta:.3f}" if decay_delta > 0.01 else " | sem decaimento"
        style_ax(ax1,
                 f"Ramp-Up P10/P50/P90  [{sigmoid_source}{decay_str}]",
                 "Semana pos-lancamento", "Unidades")
        ax1.fill_between(weeks, p10, p90, color=TEAL, alpha=0.15, label="IC P10-P90")
        ax1.plot(weeks, p50,    color=TEAL,  lw=2.5, label="P50 Forecast")
        ax1.plot(weeks, actual, color=GREEN, lw=1.8, ls="--",
                 label="Demanda Latente (obs+imputada)")
        rupt_mask = stockout_flags == 1
        if rupt_mask.any():
            ax1.scatter(weeks[rupt_mask], actual[rupt_mask],
                        color=RED, s=70, zorder=6, marker="^",
                        label=f"Ruptura imputada ({rupt_mask.sum()})")
        ax1.axvline(MIN_EVAL_WEEK-0.5, color=GOLD, lw=1.0, ls=":",
                    alpha=0.7, label=f"Inicio steady-state (sem.{MIN_EVAL_WEEK})")
        ax1.legend(fontsize=7.5, facecolor="#1e1e24", edgecolor="#2a2a30",
                   labelcolor=TEXT, loc="lower right", framealpha=0.9)
        ax1.yaxis.set_major_formatter(fmt_u)
        pw = int(p50.argmax())
        ax1.annotate("  Pico P50\n  " + f"{p50[pw]:.0f} un. | Sem.{pw+1}",
                     xy=(pw+1, p50[pw]), xytext=(pw+4, p50[pw]*0.82),
                     color=TEXT, fontsize=8,
                     arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9),
                     bbox=dict(boxstyle="round,pad=0.3", fc="#1e1e24", ec=TEAL, alpha=0.9))

        # ── 2. DADOS CENSURADOS + PICOS EXÓGENOS [IMP1] ───────────────────────
        ax2 = fig.add_subplot(gs[0,1])
        n_exog = int(sum(exog_flags)) if exog_flags is not None else 0
        style_ax(ax2,
                 f"Modulo 1 — Dados Censurados ({rupt_mask.sum()} rupt.) | Picos Exog. ({n_exog})",
                 "Semana", "Unidades")
        colors_bar = [RED if f else TEAL for f in stockout_flags]
        ax2.bar(weeks, actual, color=colors_bar, alpha=0.85, width=0.75, zorder=3)
        # Picos exógenos: marcador estrela
        if exog_flags is not None:
            ef = np.array(exog_flags)
            exog_mask = ef == 1
            if exog_mask.any():
                ax2.scatter(weeks[exog_mask], actual[exog_mask]*1.05,
                            marker="*", s=140, color=GOLD, zorder=7,
                            label=f"Pico exogeno explicado ({exog_mask.sum()})")
                # Linha P90 de referência
                p90_val = float(np.percentile(actual[actual>0], 90)) if (actual>0).any() else 0
                ax2.axhline(p90_val*P90_ANOMALY_FACTOR, color=GOLD, lw=1.0, ls="--",
                            alpha=0.6, label=f"Limiar P90x{P90_ANOMALY_FACTOR}")
        leg_el = [Patch(facecolor=TEAL, label="Venda obs. / imputada"),
                  Patch(facecolor=RED,  label="Ruptura imputada")]
        ax2.legend(handles=leg_el, fontsize=7.5, facecolor="#1e1e24",
                   edgecolor="#2a2a30", labelcolor=TEXT, framealpha=0.9)
        ax2.yaxis.set_major_formatter(fmt_u)

        # ── 3. BASS DIFFUSION ─────────────────────────────────────────────────
        ax3 = fig.add_subplot(gs[1,0])
        w = len(weeks)
        style_ax(ax3, "Bass Diffusion — Inovadores vs Imitadores",
                 "Semana", "Adotantes / semana")
        ax3.bar(weeks, innov[:w], color=TEAL, alpha=0.85, width=0.75, label="Inovadores (p=0.03)")
        ax3.bar(weeks, imit[:w], bottom=innov[:w], color=GOLD, alpha=0.75, width=0.75, label="Imitadores (q=0.38)")
        bp = int((innov[:w]+imit[:w]).argmax()) + 1
        ax3.axvline(bp, color=RED, lw=1.3, ls="--", alpha=0.8, label=f"Pico adocao: Sem.{bp}")
        ax3.legend(fontsize=7.5, facecolor="#1e1e24", edgecolor="#2a2a30",
                   labelcolor=TEXT, framealpha=0.9)
        ax3.yaxis.set_major_formatter(fmt_u)

        # ── 4. WAPE BIFÁSICO + MASE + sMAPE [AJ1][IMP2] ──────────────────────
        ax4 = fig.add_subplot(gs[1,1])
        style_ax(ax4, "Metricas — Cold Start vs Steady State", "Semana", "WAPE %")

        wape_w = np.where(actual > 0,
                          np.abs(actual - p50) / np.maximum(actual, 1e-9) * 100,
                          np.nan)
        cw = []
        for i, (v, wk) in enumerate(zip(wape_w, weeks)):
            if wk < MIN_EVAL_WEEK:
                cw.append("#555566")
            elif np.isnan(v):
                cw.append(GREEN)
            elif v > 20:
                cw.append(RED)
            elif v > 10:
                cw.append(GOLD)
            else:
                cw.append(GREEN)
        ax4.bar(weeks, np.nan_to_num(wape_w, nan=0), color=cw, alpha=0.85, width=0.75)
        ax4.axhline(20, color=RED,  lw=1.2, ls="--", label="Threshold 20%")
        ax4.axhline(10, color=GOLD, lw=1.0, ls="--", label="Threshold 10%")
        ax4.axvline(MIN_EVAL_WEEK-0.5, color=GOLD, lw=1.2, ls=":", alpha=0.7, label="Inicio steady-state")

        # Métricas completas [IMP2]
        steady_mask = weeks >= MIN_EVAL_WEEK
        cold_mask   = ~steady_mask
        wape_ss = self._wape(actual[steady_mask], p50[steady_mask])
        wape_cs = self._wape(actual[cold_mask],   p50[cold_mask])
        bias_v  = self._bias(actual[steady_mask], p50[steady_mask])
        mase_v  = self._mase(actual[steady_mask], p50[steady_mask])
        smape_v = self._smape(actual[steady_mask], p50[steady_mask])
        ic_v    = float(np.mean((actual[steady_mask]>=p10[steady_mask]) &
                                (actual[steady_mask]<=p90[steady_mask])) * 100)
        status  = ("EXCELENTE v" if not np.isnan(wape_ss) and wape_ss < 10
                   else "BOM v"  if not np.isnan(wape_ss) and wape_ss < 20
                   else "ACEITAVEL" if not np.isnan(wape_ss) and wape_ss < 30
                   else "REVISAR !")

        mase_str  = f"{mase_v:.3f}" if not np.isnan(mase_v)  else "N/A"
        smape_str = f"{smape_v:.1f}%" if not np.isnan(smape_v) else "N/A"
        metrics_txt = "\n".join([
            "[Cold Start sem.1-" + str(MIN_EVAL_WEEK-1) + "]",
            f"WAPE = {wape_cs:.1f}%  (informativo)",
            "",
            "[Steady State sem." + str(MIN_EVAL_WEEK) + "+]",
            f"WAPE  = {wape_ss:.1f}%",
            f"BIAS  = {bias_v:+.1f}%",
            f"MASE  = {mase_str}",
            f"sMAPE = {smape_str}",
            f"IC Cov = {ic_v:.0f}%",
            f"Status: {status}",
        ])
        ax4.text(0.97, 0.97, metrics_txt, transform=ax4.transAxes,
                 fontsize=8, va="top", ha="right", color=TEXT,
                 bbox=dict(boxstyle="round,pad=0.45", fc="#1e1e24", ec=TEAL, alpha=0.92))

        leg_el2 = [
            Patch(facecolor="#555566", label="Cold start (informativo)"),
            Patch(facecolor=GREEN,     label="WAPE <= 10%"),
            Patch(facecolor=GOLD,      label="WAPE 10-20%"),
            Patch(facecolor=RED,       label="WAPE > 20%"),
        ]
        ax4.legend(handles=leg_el2, fontsize=7.5, facecolor="#1e1e24",
                   edgecolor="#2a2a30", labelcolor=TEXT,
                   loc="upper left", framealpha=0.9)

        plt.tight_layout(pad=2.5)
        out = "npi_m5_resultado.png"
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        print(f"  -> Grafico salvo: {out}")


# ==============================================================================
# EXECUÇÃO PRINCIPAL
# ==============================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("NPI DEMAND FORECASTING FRAMEWORK v5.0 — M5 Walmart")
    print("=" * 70)

    sales, cal, prices = load_m5(M5_DATA_DIR)

    print("\nConvertendo para serie semanal...")
    df_weekly = melt_to_weekly(sales, cal)

    print("\nIdentificando SKUs NPI...")
    npi_ids = identify_npi_skus(df_weekly)

    target_id = None
    for candidate in npi_ids:
        s = extract_sku_rampup(df_weekly, candidate, N_NPI_WEEKS)
        if (s["weekly_sales"].sum() > 0 and
                score_rampup_quality(s["weekly_sales"].values) >= MIN_RAMPUP_SCORE):
            target_id = candidate
            break

    if target_id is None:
        print("ERRO: Nenhum SKU valido. Ajuste os parametros.")
        sys.exit(1)

    sku_series = extract_sku_rampup(df_weekly, target_id, N_NPI_WEEKS)
    print(f"\nDiagnostico do SKU:")
    diagnose_sku(sku_series, target_id)

    # MÓDULO 1
    print("\n[MODULO 1] Preprocessing & Deteccao de Ruptura...")
    prep      = StockoutPreprocessor(run_rate_window=3, global_impute_threshold=0.5)
    sku_clean = prep.fit_transform(sku_series[["week_num","weekly_sales"]])
    n_stock   = sku_clean["stockout_flag"].sum()
    print(f"  -> {n_stock} semanas de ruptura detectadas")

    # [IMP1] Cria colunas exógenas a partir do sku_series original
    exog_cols = [c for c in ["snap_CA","snap_TX","snap_WI","has_event"] if c in sku_series.columns]
    if exog_cols:
        for col in exog_cols:
            sku_clean[col] = sku_series[col].values[:len(sku_clean)]

    print("\n[MODULO 1b] Deteccao de Picos Exogenos [IMP1]...")
    sku_clean = StockoutPreprocessor.flag_exogenous_anomalies(sku_clean)
    n_exog    = int(sku_clean["exog_flag"].sum())
    print(f"  -> {n_exog} picos acima do P90 detectados")
    exog_lines = sku_clean[sku_clean["exog_flag"]==1][["week_num","exog_explanation"]]
    for _, row in exog_lines.iterrows():
        print(f"     Sem.{int(row['week_num'])}: {row['exog_explanation']}")

    # MÓDULO 2
    print("\n[MODULO 2] Cold Start - Clustering...")
    sim_eng = M5SimilarityEngine(n_clusters=8, n_similar=N_SIMILAR)
    sim_eng.build_features(df_weekly, prices)
    cands  = (df_weekly.groupby("id")["weekly_sales"].sum()
              .reset_index().query("weekly_sales > 100")["id"].tolist())
    sample = [i for i in cands if i != target_id][:500]
    rampup_db = {}
    for sid in sample:
        r = extract_sku_rampup(df_weekly, sid, N_NPI_WEEKS)
        if r["weekly_sales"].sum() > 0:
            rampup_db[sid] = r["weekly_sales"].values
    sim_eng.fit(rampup_db)
    sim_res = sim_eng.find_similar(target_id)
    print(f"  -> Top similaridade: {sim_res['top_similarity']:.3f}")
    print(f"  -> Cluster: {sim_res['target_cluster']}")

    # MÓDULO 3
    print("\n[MODULO 3] Motor de Previsao v5 (Selecao de Curvas)...")
    latent = sku_clean["latent_demand"].values
    weeks  = sku_clean["week_num"].values
    engine = DemandForecastEngine()
    engine.M = float(latent.max() * 2.0)
    print("  Comparando Sigmoid / Gompertz / Richards [IMP3]:")
    engine.fit_rampup_decay(weeks.astype(float), latent,
                             inherited_curve=sim_res.get("inherited_curve"))
    post = engine.bayesian_update(latent[:4], prior_mu=1.0)
    print(f"  -> Posterior Bayesiano: mu={post['mu']:.4f}, sigma={post['sigma']:.4f}")
    df_fc = engine.predict(n_weeks=len(weeks))
    bass  = engine.bass_diffusion(n_weeks=len(weeks))

    # MÓDULO 4
    print(f"\n[MODULO 4] Monte Carlo ({MC_SIMS:,} simulacoes)...")
    mc     = MonteCarloProbabilisticForecast(n_simulations=MC_SIMS)
    mc_res = mc.simulate(df_fc["p50"].values, demand_cv=0.12)
    print(f"  -> Pico P50: Semana {int(df_fc['p50'].argmax())+1} | {df_fc['p50'].max():,.1f} unid.")

    # MÓDULO 5
    print("\n[MODULO 5] Avaliacao v5 (WAPE + MASE + sMAPE) [IMP2]...")
    ev      = ForecastEvaluator(min_eval_week=MIN_EVAL_WEEK)
    phases  = ev.wape_by_phase(latent, mc_res["p50"], weeks)
    metrics = ev.evaluate(latent, mc_res["p10"], mc_res["p50"], mc_res["p90"], weeks)
    print(f"  -> WAPE Cold Start   (sem.1-{MIN_EVAL_WEEK-1}): {phases['WAPE_coldstart_%']}%  [informativo]")
    print(f"  -> WAPE Steady State (sem.{MIN_EVAL_WEEK}+)  : {phases['WAPE_steadystate_%']}%  [metrica principal]")
    for k, v in metrics.items():
        print(f"  -> {k}: {v}")

    print("\nGerando visualizacoes...")
    ev.plot_all(
        weeks=weeks, actual=latent,
        p10=mc_res["p10"], p50=mc_res["p50"], p90=mc_res["p90"],
        innov=bass["innovators"], imit=bass["imitators"],
        stockout_flags=sku_clean["stockout_flag"].values,
        exog_flags=sku_clean["exog_flag"].values,
        exog_labels=sku_clean["exog_explanation"].values,
        item_id=target_id,
        sigmoid_source=engine.sigmoid_params.get("source",""),
        decay_delta=engine.decay_delta,
    )

    print("\n>>> CONCLUIDO! Arquivo: npi_m5_resultado.png")
    print("=" * 70)
