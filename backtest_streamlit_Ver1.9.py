import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler          # ⑦ 피처 스케일링
from sklearn.model_selection import cross_val_score     # ⑧ 하이퍼파라미터 교차검증
import matplotlib.pyplot as plt
import requests
from io import StringIO
import quantstats as qs
import plotly.express as px
import seaborn as sns
from datetime import datetime, timedelta
import concurrent.futures
import time
import random
import os

st.set_page_config(page_title="Advanced AI Quant Lab", layout="wide")

# ─────────────────────────────────────────────
# 환경 감지 — 로컬 vs Streamlit Cloud
# ─────────────────────────────────────────────
IS_CLOUD     = os.environ.get("STREAMLIT_SERVER_HEADLESS", "0") == "1"
MAX_WORKERS  = 2          if IS_CLOUD else 8
SLEEP_JITTER = (2.0, 4.0) if IS_CLOUD else (0.5, 1.5)
RETRY_SLEEP  = (4.0, 7.0) if IS_CLOUD else (2.0, 3.5)

# ④ 거래비용 상수 (편도 0.08% = 왕복 0.16%)
TRANSACTION_COST = 0.0008


# ─────────────────────────────────────────────
# S&P 500 종목 리스트
# ─────────────────────────────────────────────
@st.cache_data
def get_sp500_info():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    df = pd.read_html(StringIO(response.text))[0]
    sectors = sorted(df["GICS Sector"].unique().tolist())
    return df, sectors


# ─────────────────────────────────────────────
# 단일 티커 재무 fetch
# ─────────────────────────────────────────────
def _fetch_one_ticker(ticker: str) -> tuple[str, dict | None]:
    MAX_RETRIES = 3
    for attempt in range(MAX_RETRIES):
        try:
            tk   = yf.Ticker(ticker)
            info = tk.info or {}
            if not info or len(info) < 5:
                raise ValueError("info 응답 비어 있음")

            def _safe_T(df) -> pd.DataFrame:
                if df is None or (hasattr(df, "empty") and df.empty):
                    return pd.DataFrame()
                return df.T

            q_fin = tk.quarterly_financials
            a_fin = tk.financials
            if (q_fin is None or q_fin.empty) and (a_fin is None or a_fin.empty):
                return ticker, None

            return ticker, {
                "q_fin": _safe_T(q_fin),
                "q_bal": _safe_T(tk.quarterly_balance_sheet),
                "q_cf":  _safe_T(tk.quarterly_cashflow),
                "a_fin": _safe_T(a_fin),
                "a_bal": _safe_T(tk.balance_sheet),
                "a_cf":  _safe_T(tk.cashflow),
                "info":  info,
            }
        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep((2 ** (attempt + 1)) + random.uniform(*SLEEP_JITTER))
    return ticker, None


# ─────────────────────────────────────────────
# 재무 데이터 수집
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def get_all_financial_source(tickers: list) -> dict:
    data_cache: dict = {}
    failed:     list = []
    progress_bar = st.progress(0, text="재무 데이터 수집 중...")
    total = len(tickers)

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {executor.submit(_fetch_one_ticker, t): t for t in tickers}
        done_count = 0
        try:
            for future in concurrent.futures.as_completed(future_map, timeout=300):
                t = future_map[future]
                try:
                    ticker, result = future.result(timeout=45)
                    if result:
                        data_cache[ticker] = result
                    else:
                        failed.append(ticker)
                except Exception:
                    failed.append(t)
                done_count += 1
                progress_bar.progress(
                    int(done_count / total * 80),
                    text=f"재무 데이터 수집 중... ({done_count}/{total})"
                )
        except concurrent.futures.TimeoutError:
            for f, t in future_map.items():
                if not f.done():
                    failed.append(t)
                    f.cancel()

    if failed:
        for idx, ticker in enumerate(failed):
            progress_bar.progress(
                int(80 + idx / max(len(failed), 1) * 18),
                text=f"재시도 중... ({idx+1}/{len(failed)})"
            )
            time.sleep(random.uniform(*RETRY_SLEEP))
            _, result = _fetch_one_ticker(ticker)
            if result:
                data_cache[ticker] = result

    progress_bar.progress(100, text=f"완료: {len(data_cache)}/{total}개 수집")
    time.sleep(0.5)
    progress_bar.empty()
    return data_cache


# ─────────────────────────────────────────────
# 헬퍼: Series 안전 추출
# ─────────────────────────────────────────────
def _safe_get(series: pd.Series, key: str, default=0):
    val = series.get(key, default)
    if val is None:
        return default
    if isinstance(val, float) and np.isnan(val):
        return default
    return val


# ─────────────────────────────────────────────
# 헬퍼: Point-in-Time 재무 행 추출
#  ① 반환 시 is_quarterly 플래그도 함께 반환
#     → 연간 데이터 폴백 시 ×4 방지에 사용
# ─────────────────────────────────────────────
def _get_pit_row(
    src: dict, q_key: str, a_key: str, ref_dt: pd.Timestamp
) -> tuple[pd.Series, bool]:
    """
    반환: (row: pd.Series, is_quarterly: bool)
      is_quarterly=True  → 분기 데이터 사용 → 연간화 시 ×4
      is_quarterly=False → 연간 데이터 사용 → 연간화 시 ×1
    """
    q_df = src.get(q_key, pd.DataFrame())
    a_df = src.get(a_key, pd.DataFrame())
    empty = pd.Series(dtype=float)

    def _best_row(df: pd.DataFrame) -> pd.Series | None:
        if df.empty:
            return None
        valid = df[df.index <= (ref_dt + pd.Timedelta(days=45))]
        if valid.empty:
            return None
        idx_min = int(np.abs((valid.index - ref_dt).days).argmin())
        row = valid.iloc[idx_min].copy()
        if row.isna().sum() > len(row) * 0.5:
            filled = df.ffill().bfill()
            if not filled.empty and idx_min < len(filled):
                row = filled.iloc[idx_min]
        return row

    # 분기 우선
    row = _best_row(q_df)
    if row is not None:
        return row, True

    # 연간 폴백
    row = _best_row(a_df)
    if row is not None:
        return row, False

    return empty, False


# ─────────────────────────────────────────────
# ② 성장률 계산 — 분기/연간 타입 통일
#    분기: QoQ(전분기 대비), 연간: YoY(전년 대비)
#    → 같은 타입끼리만 비교
# ─────────────────────────────────────────────
def _get_prev_row(
    src: dict, q_key: str, a_key: str,
    ref_dt: pd.Timestamp, is_quarterly: bool
) -> pd.Series:
    key   = q_key if is_quarterly else a_key
    df    = src.get(key, pd.DataFrame())
    empty = pd.Series(dtype=float)
    if df.empty:
        return empty
    valid = df[df.index <= (ref_dt + pd.Timedelta(days=45))]
    if len(valid) < 2:
        return empty
    return valid.iloc[1]   # 현재 바로 직전 행


# ─────────────────────────────────────────────
# ⑧ 워크포워드용 최적 모델 선택
#    max_depth 후보 3가지를 교차검증으로 비교
# ─────────────────────────────────────────────
def _select_best_model(X: pd.DataFrame, y: pd.Series) -> RandomForestRegressor:
    best_score = -np.inf
    best_depth = 5
    for depth in [5, 10, 15]:
        model = RandomForestRegressor(
            n_estimators=100, random_state=42,
            max_depth=depth, min_samples_leaf=5, n_jobs=-1
        )
        # 데이터가 너무 적으면 cv=2, 충분하면 cv=3
        cv = 2 if len(X) < 30 else 3
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
            mean_score = scores.mean()
        except Exception:
            mean_score = -np.inf
        if mean_score > best_score:
            best_score = mean_score
            best_depth = depth

    best_model = RandomForestRegressor(
        n_estimators=100, random_state=42,
        max_depth=best_depth, min_samples_leaf=5, n_jobs=-1
    )
    best_model.fit(X, y)
    return best_model


# ─────────────────────────────────────────────
# ML 피처 추출 (Point-in-Time)
# ─────────────────────────────────────────────
def fetch_ml_data_optimized_pit(
    tickers, ref_date, full_hist_data, source_cache, is_training=True
):
    features_list = []
    ref_dt        = pd.to_datetime(ref_date)
    all_level0    = set(full_hist_data.columns.get_level_values(0))

    for ticker in tickers:
        try:
            if ticker not in all_level0:
                continue

            ticker_prices = full_hist_data[ticker].dropna(how="all")
            hist = ticker_prices[ticker_prices.index < ref_dt].tail(252)
            if len(hist) < 252:
                continue

            close_now = float(hist["Close"].iloc[-1])
            src       = source_cache.get(ticker, {})
            info      = src.get("info", {})

            # ① is_quarterly 플래그 수령
            cur, is_q_fin = _get_pit_row(src, "q_fin", "a_fin", ref_dt)
            bal, is_q_bal = _get_pit_row(src, "q_bal", "a_bal", ref_dt)
            cf,  is_q_cf  = _get_pit_row(src, "q_cf",  "a_cf",  ref_dt)

            # ① 연간화 계수: 분기=4, 연간=1
            fin_mult = 4 if is_q_fin else 1
            cf_mult  = 4 if is_q_cf  else 1

            # ② 동일 타입 직전 행으로 성장률 계산
            prev = _get_prev_row(src, "q_fin", "a_fin", ref_dt, is_q_fin)

            shares        = float(info.get("sharesOutstanding") or 1)
            mkt_cap       = close_now * shares
            net_income    = _safe_get(cur, "Net Income")
            revenue       = _safe_get(cur, "Total Revenue")
            gross_profit  = _safe_get(cur, "Gross Profit")
            total_assets  = _safe_get(bal, "Total Assets",  1) or 1
            fcf           = _safe_get(cf,  "Free Cash Flow")
            ebit          = _safe_get(cur, "EBIT")
            da            = _safe_get(cf,  "Depreciation And Amortization")
            ebitda        = ebit + da
            # ① 연간화 시 올바른 배수 적용
            annual_ni     = net_income * fin_mult
            annual_rev    = revenue    * fin_mult
            annual_ebitda = ebitda     * fin_mult
            annual_fcf    = fcf        * cf_mult
            total_debt    = _safe_get(bal, "Total Debt")
            cash_eq       = _safe_get(bal, "Cash And Cash Equivalents")
            equity        = _safe_get(bal, "Stockholders Equity", 1) or 1
            ev            = mkt_cap + total_debt - cash_eq

            # ② 동일 타입 직전 데이터로 성장률 계산
            prev_revenue    = _safe_get(prev, "Total Revenue", 1) or 1
            prev_net_income = _safe_get(prev, "Net Income",    1) or 1

            close = hist["Close"]

            data = {
                "Ticker":             ticker,
                # ─ 밸류에이션 (연간화 배수 수정) ─
                "P/E":                mkt_cap / annual_ni        if annual_ni > 0        else 0,
                "P/S":                mkt_cap / annual_rev       if annual_rev > 0       else 0,
                "P/B":                mkt_cap / equity,
                "P/FCF":              mkt_cap / annual_fcf       if annual_fcf > 0       else 0,
                "EV/EBITDA":          ev / annual_ebitda         if annual_ebitda > 0    else 0,
                "FCF_Yield":          annual_fcf / mkt_cap       if mkt_cap > 0          else 0,
                # ─ 수익성 ─
                "ROE":                net_income / equity,
                "ROA":                net_income / total_assets,
                "Gross_Margin":       gross_profit / revenue     if revenue > 0          else 0,
                "Operating_Margin":   ebit / revenue             if revenue > 0          else 0,
                "EBITDA_Margin":      ebitda / revenue           if revenue > 0          else 0,
                "GP_A_Quality":       gross_profit / total_assets,
                "Asset_Turnover":     revenue / total_assets,
                "Inventory_Turnover": (
                    _safe_get(cur, "Cost Of Revenue") / (_safe_get(bal, "Inventory", 1) or 1)
                    if "Inventory" in bal.index else 0
                ),
                # ─ 성장률 (② 동일 타입 직전 비교) ─
                "Revenue_Growth":     (revenue / prev_revenue) - 1,
                "NetIncome_Growth":   (net_income / prev_net_income) - 1,
                # ─ 재무 건전성 ─
                "Debt_Equity":        total_debt / equity,
                "Current_Ratio":      (
                    _safe_get(bal, "Total Current Assets")
                    / (_safe_get(bal, "Total Current Liabilities", 1) or 1)
                    if "Total Current Liabilities" in bal.index else 0
                ),
                "Interest_Coverage":  (
                    ebit / (_safe_get(cur, "Interest Expense", 1) or 1)
                    if "Interest Expense" in cur.index else 0
                ),
                # ─ 모멘텀 ─
                "Mom_1w":             (close_now / close.iloc[-6])   - 1 if len(hist) >= 6   else 0,
                "Mom_1m":             (close_now / close.iloc[-21])  - 1 if len(hist) >= 21  else 0,
                "Mom_6m":             (close_now / close.iloc[-127]) - 1 if len(hist) >= 127 else 0,
                "Mom_12m":            (close_now / close.iloc[-252]) - 1 if len(hist) >= 252 else 0,
                "MA_Convergence":     (
                    (close.rolling(20).mean().iloc[-1] / close.rolling(200).mean().iloc[-1]) - 1
                    if len(hist) >= 200 else 0
                ),
                "MA50_Dist":          close_now / close.rolling(50).mean().iloc[-1]  if len(hist) >= 50  else 1,
                "MA200_Dist":         close_now / close.rolling(200).mean().iloc[-1] if len(hist) >= 200 else 1,
                "Momentum_12M_1M":    (close.iloc[-21] / close.iloc[-252]) - 1 if len(hist) >= 252 else 0,
                "Momentum_6M_1M":     (close.iloc[-21] / close.iloc[-126]) - 1 if len(hist) >= 126 else 0,
                "Momentum_Custom":    (close.iloc[-1]  / close.iloc[-63])  - 1 if len(hist) >= 63  else 0,
                # ─ 변동성/거래량 ─
                "Volatility_30d":     close.pct_change().std() * np.sqrt(252),
                "Risk_Adj_Return":    (
                    close.pct_change().mean() / close.pct_change().std()
                    if close.pct_change().std() != 0 else 0
                ),
                "Vol_Change":         (
                    hist["Volume"].iloc[-1] / hist["Volume"].rolling(21).mean().iloc[-1]
                    if len(hist) >= 21 else 1
                ),
            }

            if is_training:
                future_prices = ticker_prices[ticker_prices.index >= ref_dt].head(22)
                if len(future_prices) >= 20:
                    data["Target_Return"] = (
                        future_prices["Close"].iloc[-1] / future_prices["Close"].iloc[0]
                    ) - 1
                else:
                    continue

            features_list.append(data)

        except Exception:
            continue

    return pd.DataFrame(features_list).replace([np.inf, -np.inf], np.nan).fillna(0)


# ─────────────────────────────────────────────
# 리밸런싱 메타 정보 수집
# ─────────────────────────────────────────────
def get_rebalance_meta(tickers, ref_date, full_hist_data, source_cache):
    ref_dt     = pd.to_datetime(ref_date)
    all_level0 = set(full_hist_data.columns.get_level_values(0))
    period_starts = []
    data_types: dict = {}

    for ticker in tickers:
        if ticker not in all_level0:
            continue
        ticker_prices = full_hist_data[ticker].dropna(how="all")
        hist = ticker_prices[ticker_prices.index < ref_dt].tail(252)
        if len(hist) < 252:
            continue
        period_starts.append(hist.index[0])

        src   = source_cache.get(ticker, {})
        q_fin = src.get("q_fin", pd.DataFrame())
        a_fin = src.get("a_fin", pd.DataFrame())
        q_valid = q_fin[q_fin.index <= (ref_dt + pd.Timedelta(days=45))] if not q_fin.empty else pd.DataFrame()
        a_valid = a_fin[a_fin.index <= (ref_dt + pd.Timedelta(days=45))] if not a_fin.empty else pd.DataFrame()

        if not q_valid.empty:
            data_types[ticker] = "분기"
        elif not a_valid.empty:
            data_types[ticker] = "연간(분기 대체)"
        else:
            data_types[ticker] = "없음"

    period_start = min(period_starts).strftime("%Y-%m-%d") if period_starts else "N/A"
    total   = len(data_types)
    q_count = sum(1 for v in data_types.values() if v == "분기")
    a_count = sum(1 for v in data_types.values() if v == "연간(분기 대체)")
    n_count = sum(1 for v in data_types.values() if v == "없음")

    return {
        "period_start": period_start,
        "period_end":   ref_dt.strftime("%Y-%m-%d"),
        "data_types":   data_types,
        "q_count": q_count, "a_count": a_count,
        "n_count": n_count, "total":   total,
    }


# ─────────────────────────────────────────────
# ⑥ 성과 지표 계산 (확장)
# ─────────────────────────────────────────────
def get_extended_metrics(rets: pd.Series, label: str) -> dict:
    cum  = (1 + rets).prod() - 1
    yrs  = max((rets.index[-1] - rets.index[0]).days / 365.25, 0.1)
    cagr = (1 + cum) ** (1 / yrs) - 1
    mdd  = qs.stats.max_drawdown(rets)

    # 월별 승률
    monthly = rets.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    win_rate = (monthly > 0).mean() if len(monthly) > 0 else 0

    # 최대 연속 손실 일수
    losing = (rets < 0).astype(int)
    max_streak = 0
    streak = 0
    for v in losing:
        streak = streak + 1 if v else 0
        max_streak = max(max_streak, streak)

    return {
        "누적 수익률":        f"{cum * 100:.2f}%",
        "연수익률(CAGR)":     f"{cagr * 100:.2f}%",
        "샤프 지수":          round(qs.stats.sharpe(rets), 2),
        "칼마 비율":          round(cagr / abs(mdd), 2) if mdd != 0 else "∞",
        "MDD":                f"{mdd * 100:.2f}%",
        "월별 승률":          f"{win_rate * 100:.1f}%",
        "최대 연속 손실(일)": max_streak,
    }


# ─────────────────────────────────────────────
# 지표 중요도 히트맵
# ─────────────────────────────────────────────
def display_importance_heatmap(imp_all_df):
    st.subheader("🌡️ 지표별 영향력 타임라인 (Heatmap)")
    if imp_all_df.empty:
        st.write("데이터가 부족합니다.")
        return
    top_15 = imp_all_df.mean().nlargest(15).index.tolist()
    heatmap_data = imp_all_df[top_15].T
    try:
        heatmap_data.columns = [pd.to_datetime(d).strftime("%Y-%m") for d in heatmap_data.columns]
    except Exception:
        heatmap_data.columns = [str(d)[:7] for d in heatmap_data.columns]
    heatmap_norm = heatmap_data.apply(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9), axis=0
    )
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        heatmap_norm, annot=heatmap_data.values, fmt=".2f",
        cmap="YlGnBu", linewidths=0.5,
        cbar_kws={"label": "Relative Importance"}, ax=ax,
    )
    plt.title("Feature Importance Heatmap (Top 15 Metrics)")
    plt.xticks(rotation=45)
    st.pyplot(fig)


# ─────────────────────────────────────────────
# 리밸런싱 주기 → 권장 백테스트 기간 자동 계산
# 기준: 최소 30구간 확보 + yfinance 재무 데이터 한계(4~5년) 고려
# ─────────────────────────────────────────────
RECOMMENDED_YEARS = {1: 4, 3: 5, 6: 7, 12: 10}   # 주기(개월) → 권장 백테스트 연수
MIN_PERIODS       = {1: 48, 3: 20, 6: 14, 12: 10} # 최소 구간 수

def get_recommended_dates(reb_months: int) -> tuple[datetime, datetime]:
    years     = RECOMMENDED_YEARS.get(reb_months, 5)
    end_dt    = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_dt  = end_dt - timedelta(days=int(years * 365.25))
    return start_dt, end_dt


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.title("🚀 Advanced AI Quant Lab")
df_sp500, all_sectors = get_sp500_info()

with st.expander("🛠 전략 설정 및 필터링", expanded=True):

    # ── 행 1: 섹터 / 종목 수 / 비중 방식 ─────────────────────
    r1c1, r1c2, r1c3 = st.columns([3, 1, 2])
    with r1c1:
        selected_sectors = st.multiselect(
            "1. 분석 섹터 선택", all_sectors, default=["Information Technology"]
        )
    with r1c2:
        top_n = st.number_input("2. 선정 종목 수", min_value=3, max_value=20, value=5)
    with r1c3:
        weight_mode = st.radio(
            "3. 포트폴리오 비중 방식", ["동일 비중", "AI 점수 비례"], horizontal=True
        )

    st.divider()

    # ── 행 2: 리밸런싱 주기 (먼저 선택 → 날짜 자동 세팅) ─────
    reb_months = st.select_slider(
        "4. 리밸런싱 주기 (개월)",
        options=[1, 3, 6, 12], value=1,
        help="주기를 바꾸면 아래 백테스트 기간이 자동으로 추천값으로 바뀌어요."
    )

    auto_start, auto_end = get_recommended_dates(reb_months)
    min_periods_needed   = MIN_PERIODS[reb_months]
    years_needed         = RECOMMENDED_YEARS[reb_months]

    st.caption(
        f"📌 {reb_months}개월 리밸런싱 권장: **{years_needed}년** 백테스트 "
        f"(최소 {min_periods_needed}구간) — yfinance 재무 데이터 한계 고려한 추천값이에요."
    )

    # ── 행 3: 날짜 (자동 세팅 + 수동 수정 가능) ──────────────
    use_auto = st.checkbox("✅ 권장 기간 자동 적용", value=True)

    d1, d2 = st.columns(2)
    with d1:
        if use_auto:
            start_date = st.date_input(
                "5. 백테스트 시작일 (자동)", value=auto_start,
                disabled=True
            )
            start_date = auto_start.date()
        else:
            start_date = st.date_input(
                "5. 백테스트 시작일 (수동 입력)", value=auto_start
            )
    with d2:
        if use_auto:
            end_date = st.date_input(
                "6. 백테스트 종료일 (자동)", value=auto_end,
                disabled=True
            )
            end_date = auto_end.date()
        else:
            end_date = st.date_input(
                "6. 백테스트 종료일 (수동 입력)", value=auto_end
            )

    # 실제 구간 수 계산 및 경고
    actual_periods = max(
        int((pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
            / (reb_months * 30.44)), 0
    )
    if actual_periods < min_periods_needed:
        st.warning(
            f"⚠️ 현재 설정으로 예상 구간 수: **{actual_periods}개** — "
            f"통계적 신뢰도를 위해 최소 {min_periods_needed}구간을 권장해요."
        )
    else:
        st.success(f"✅ 예상 구간 수: **{actual_periods}개** ({years_needed}년치)")

    st.divider()

    # ── 행 4: 거래비용 ────────────────────────────────────────
    tc_bps = st.number_input(
        "7. 거래비용 (편도, bps)",
        min_value=0, max_value=50,
        value=10 if reb_months == 1 else 8,
        help=(
            "1bps = 0.01%. 유지 종목은 비용 없음, 교체 종목에만 편도 비용 적용.\n"
            "1개월 리밸런싱은 교체 빈도가 높아 10bps 권장, 3개월은 8bps."
        )
    )

    run_analysis = st.button("백테스트 실행 🚀", use_container_width=True)

TRANSACTION_COST = tc_bps / 10000 if run_analysis else TRANSACTION_COST


# ─────────────────────────────────────────────
# 백테스트 실행
# ─────────────────────────────────────────────
if run_analysis:
    tickers = (
        df_sp500[df_sp500["GICS Sector"].isin(selected_sectors)]["Symbol"]
        .str.replace(".", "-", regex=False)
        .tolist()
    )

    st.info(f"📡 주가 데이터 다운로드 중... ({len(tickers)}개 티커)")
    full_hist_data = yf.download(
        tickers,
        start=pd.to_datetime(start_date) - timedelta(days=400),
        end=pd.to_datetime(end_date) + timedelta(days=40),
        group_by="ticker",
        progress=False,
        threads=True,
    )
    st.success(f"✅ 주가 데이터 완료 ({full_hist_data.shape[1]}개 컬럼)")

    st.info("📊 재무 데이터 수집 중...")
    source_cache = get_all_financial_source(tickers)
    fin_count    = len(source_cache)
    st.success(f"✅ 재무 데이터 완료: {fin_count}/{len(tickers)}개")
    if fin_count < len(tickers) * 0.5:
        st.warning("⚠️ 재무 데이터 수집률이 낮아요. 잠시 후 다시 시도해보세요.")

    date_range           = pd.date_range(start=start_date, end=end_date, freq=f"{reb_months}MS")
    all_strategy_returns = pd.Series(dtype=float)
    rebalance_details    = []
    importance_history   = []
    final_model_columns  = None
    latest_model         = None
    scaler               = RobustScaler()   # ⑦ 스케일러 초기화

    # ── 초기 모델 학습 ────────────────────────────────────────
    with st.status("🏗️ 초기 모델 학습 중...", expanded=True) as status:
        train_df_init = fetch_ml_data_optimized_pit(
            tickers, date_range[0], full_hist_data, source_cache, is_training=True
        )
        st.write(f"학습 데이터: {len(train_df_init)}개 종목")

        if not train_df_init.empty:
            fin_cols   = ["P/E", "P/S", "ROE", "ROA", "Gross_Margin"]
            zero_ratio = (train_df_init[fin_cols] == 0).all(axis=1).mean()
            st.write(f"재무 지표 정상 비율: {(1 - zero_ratio) * 100:.0f}%")

        if not train_df_init.empty and "Target_Return" in train_df_init.columns:
            X_init = train_df_init.drop(["Ticker", "Target_Return"], axis=1)
            y_init = train_df_init["Target_Return"]
            final_model_columns = X_init.columns.tolist()

            # ⑦ 초기 스케일러 fit
            X_init_scaled = pd.DataFrame(
                scaler.fit_transform(X_init),
                columns=final_model_columns
            )
            # ⑧ 교차검증으로 최적 모델 선택
            latest_model = _select_best_model(X_init_scaled, y_init)
            status.update(
                label=f"✅ 초기 학습 완료 (depth={latest_model.max_depth}, "
                      f"기준일: {date_range[0].strftime('%Y-%m-%d')})",
                state="complete"
            )
        else:
            status.update(label="❌ 학습 데이터 없음", state="error")

    if latest_model is None or final_model_columns is None:
        st.error("초기 학습 데이터를 가져오지 못했습니다.")
        st.stop()

    # ── 워크포워드 루프 ───────────────────────────────────────
    # ③ 올바른 워크포워드 구조
    #    i번째 루프:
    #      - curr_reb 시점 피처로 종목 추론  (모델: i-1 구간 학습)
    #      - curr_reb ~ next_reb 실제 수익률 기록
    #      - next_reb 이후: curr_reb 피처 + 실제 수익률(curr~next)로 재학습
    #        → 이 모델이 i+1번째 루프의 추론에 사용됨
    loop_bar    = st.progress(0, text="워크포워드 진행 중...")
    total_steps = len(date_range) - 1
    prev_tickers: set = set()   # 직전 구간 보유 종목 추적 (교체율 계산용)

    for i in range(total_steps):
        curr_reb = date_range[i]
        next_reb = date_range[i + 1]
        loop_bar.progress(
            int(i / total_steps * 100),
            text=f"리밸런싱 {i+1}/{total_steps} ({curr_reb.strftime('%Y-%m')})"
        )

        # ① curr_reb 시점 피처로 추론
        inference_df = fetch_ml_data_optimized_pit(
            tickers, curr_reb, full_hist_data, source_cache, is_training=False
        )
        if inference_df.empty:
            continue

        X_infer = (
            inference_df.drop(["Ticker"], axis=1)
            .reindex(columns=final_model_columns)
            .fillna(0)
        )
        # ⑦ 스케일링 적용 (transform만, fit은 하지 않음)
        X_infer_scaled = pd.DataFrame(
            scaler.transform(X_infer),
            columns=final_model_columns
        )
        inference_df = inference_df.copy()
        inference_df["Prediction"] = latest_model.predict(X_infer_scaled)

        selected_rows = inference_df.nlargest(top_n, "Prediction").copy()
        sel_tickers   = selected_rows["Ticker"].tolist()
        predictions   = selected_rows["Prediction"].values

        importances = pd.Series(latest_model.feature_importances_, index=final_model_columns)
        importance_history.append({"Date": curr_reb.strftime("%Y-%m-%d"), **importances.to_dict()})
        meta = get_rebalance_meta(tickers, curr_reb, full_hist_data, source_cache)
        rebalance_details.append({
            "date":          curr_reb.strftime("%Y-%m-%d"),
            "next_date":     next_reb.strftime("%Y-%m-%d"),
            "selected_data": selected_rows,
            "importance":    importances,
            "meta":          meta,
            "best_depth":    latest_model.max_depth,
        })

        # ⑤ 비중 계산 + 교체율 기반 거래비용
        valid_sel  = [t for t in sel_tickers if t in full_hist_data.columns.get_level_values(0)]
        curr_set   = set(valid_sel)

        # 교체율 계산: 이전 구간 대비 새로 매수/매도하는 종목 비율
        if prev_tickers:
            new_buys  = curr_set - prev_tickers
            new_sells = prev_tickers - curr_set
            # 교체 종목 수 / 전체 보유 종목 수 = 교체율
            turnover  = len(new_buys) / max(len(curr_set), 1)
        else:
            # 첫 리밸런싱: 전액 신규 매수
            turnover = 1.0

        # 실제 거래비용 = 편도 비용 × 교체율 × 2 (매수 + 매도)
        actual_tc = TRANSACTION_COST * turnover * 2

        # rebalance_details에 교체율 정보 추가
        rebalance_details[-1]["turnover"]    = turnover
        rebalance_details[-1]["new_buys"]    = sorted(curr_set - prev_tickers)
        rebalance_details[-1]["new_sells"]   = sorted(prev_tickers - curr_set)
        rebalance_details[-1]["kept"]        = sorted(curr_set & prev_tickers)
        rebalance_details[-1]["actual_tc"]   = actual_tc

        prev_tickers = curr_set   # 다음 루프를 위해 저장

        if valid_sel:
            subset = full_hist_data[valid_sel].loc[curr_reb:next_reb]
            if not subset.empty:
                try:
                    test_prices = subset.xs("Close", axis=1, level=1)
                    stock_rets  = test_prices.pct_change().fillna(0)

                    if weight_mode == "AI 점수 비례":
                        valid_preds = np.array([
                            predictions[sel_tickers.index(t)]
                            for t in valid_sel
                        ])
                        exp_p   = np.exp(valid_preds - valid_preds.max())
                        weights = exp_p / exp_p.sum()
                    else:
                        weights = np.ones(len(valid_sel)) / len(valid_sel)

                    weighted_rets = stock_rets.values @ weights
                    period_rets   = pd.Series(weighted_rets, index=stock_rets.index)

                    # 교체율 기반 거래비용 반영 (첫날에 일괄 차감)
                    period_rets.iloc[0] -= actual_tc

                    all_strategy_returns = pd.concat([all_strategy_returns, period_rets])
                except Exception:
                    pass

        # ③ 진짜 워크포워드 재학습
        #    curr_reb 피처 + curr_reb~next_reb 사이 실제 수익률을 Target으로 사용
        train_df_wf = fetch_ml_data_optimized_pit(
            tickers, curr_reb, full_hist_data, source_cache, is_training=True
        )
        if not train_df_wf.empty and "Target_Return" in train_df_wf.columns:
            X_wf = train_df_wf.drop(["Ticker", "Target_Return"], axis=1)
            y_wf = train_df_wf["Target_Return"]
            # ⑦ 재학습 시 스케일러도 갱신
            X_wf_scaled = pd.DataFrame(
                scaler.fit_transform(X_wf),
                columns=final_model_columns
            )
            # ⑧ 매 구간마다 최적 depth 재선택
            latest_model = _select_best_model(X_wf_scaled, y_wf)

    loop_bar.progress(100, text="워크포워드 완료!")

    # ─────────────────────────────────────────────
    # 결과 시각화
    # ─────────────────────────────────────────────
    if all_strategy_returns.empty:
        st.warning("수익률 데이터를 생성하지 못했습니다.")
        st.stop()

    all_strategy_returns = (
        all_strategy_returns[~all_strategy_returns.index.duplicated()].sort_index()
    )
    start_ts, end_ts = all_strategy_returns.index[0], all_strategy_returns.index[-1]

    bench_raw = yf.download(
        ["SPY", "QQQ", "TQQQ"],
        start=start_ts - timedelta(days=5),
        end=end_ts + timedelta(days=5),
        progress=False,
    )["Close"]
    bench_raw = bench_raw.ffill().reindex(all_strategy_returns.index).ffill()

    spy_rets  = bench_raw["SPY"].pct_change().fillna(0)
    qqq_rets  = bench_raw["QQQ"].pct_change().fillna(0)
    tqqq_rets = bench_raw["TQQQ"].pct_change().fillna(0)

    st.header(f"📊 {', '.join(selected_sectors)} 전략 성과 보고서")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📈 누적 수익률 비교")
        cum_returns = pd.DataFrame({
            "Strategy (AI)": (1 + all_strategy_returns).cumprod(),
            "SPY":   (1 + spy_rets).cumprod(),
            "QQQ":   (1 + qqq_rets).cumprod(),
            "TQQQ":  (1 + tqqq_rets).cumprod(),
        })
        fig_cum = px.line(
            cum_returns, x=cum_returns.index, y=cum_returns.columns,
            color_discrete_map={
                "Strategy (AI)": "firebrick",
                "SPY": "royalblue", "QQQ": "seagreen", "TQQQ": "orange",
            },
        )
        fig_cum.update_layout(
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500,
        )
        st.plotly_chart(fig_cum, use_container_width=True)

    with col2:
        # ⑥ 확장된 성과 지표 테이블
        st.subheader("💡 상세 성과 지표")
        metrics_df = pd.DataFrame({
            "AI Strategy": get_extended_metrics(all_strategy_returns, "AI"),
            "SPY":         get_extended_metrics(spy_rets,  "SPY"),
            "QQQ":         get_extended_metrics(qqq_rets,  "QQQ"),
            "TQQQ":        get_extended_metrics(tqqq_rets, "TQQQ"),
        })
        st.table(metrics_df)

        # ⑥ 알파/베타 (vs SPY)
        try:
            aligned = pd.concat(
                [all_strategy_returns, spy_rets], axis=1, join="inner"
            ).dropna()
            aligned.columns = ["strat", "spy"]
            beta  = aligned.cov().iloc[0, 1] / aligned["spy"].var()
            alpha = (aligned["strat"].mean() - beta * aligned["spy"].mean()) * 252
            st.markdown(f"**vs SPY —** Alpha: `{alpha*100:.2f}%` / Beta: `{beta:.2f}`")
        except Exception:
            pass

    # ── 리밸런싱 히스토리 ─────────────────────────────────────
    st.divider()
    st.subheader("🗓️ 리밸런싱 히스토리")

    for detail in reversed(rebalance_details):
        meta       = detail.get("meta", {})
        check_cols = [c for c in ["P/E", "ROE"] if c in detail["selected_data"].columns]
        has_fin    = (
            not (detail["selected_data"][check_cols].abs() < 0.0001).all().all()
            if check_cols else False
        )
        q_count = meta.get("q_count", 0)
        a_count = meta.get("a_count", 0)
        n_count = meta.get("n_count", 0)
        total   = meta.get("total", 1) or 1

        if q_count / total >= 0.7:
            data_badge = "🟢 분기 데이터"
        elif a_count / total >= 0.5:
            data_badge = "🟡 연간(분기 대체)"
        else:
            data_badge = "🔴 데이터 부족"

        fin_badge    = "✅ 재무+가격" if has_fin else "⚠️ 모멘텀 중심"
        period_start = meta.get("period_start", "N/A")
        period_end   = meta.get("period_end", detail["date"])
        next_date    = detail.get("next_date", "N/A")
        best_depth   = detail.get("best_depth", "-")

        with st.expander(
            f"📅 {detail['date']}  |  {fin_badge}  |  {data_badge}",
            expanded=False
        ):
            m1, m2, m3, m4 = st.columns(4)
            m1.markdown(f"📆 **리밸런싱 기준일**<br>{detail['date']}", unsafe_allow_html=True)
            m2.markdown(f"📈 **보유 기간**<br>{period_end} ~ {next_date}", unsafe_allow_html=True)
            m3.markdown(f"📊 **주가 학습 기간**<br>{period_start} ~ {period_end}", unsafe_allow_html=True)
            m4.markdown(
                f"🗃️ **재무 데이터**<br>분기 {q_count}개 / 연간 {a_count}개 / 없음 {n_count}개"
                f"<br>🌲 모델 depth: {best_depth}",
                unsafe_allow_html=True
            )
            st.divider()

            # ── 교체율 및 거래비용 정보 ──────────────────────────
            turnover  = detail.get("turnover", 0)
            new_buys  = detail.get("new_buys",  [])
            new_sells = detail.get("new_sells", [])
            kept      = detail.get("kept",      [])
            actual_tc = detail.get("actual_tc", 0)

            tc1, tc2, tc3, tc4 = st.columns(4)
            tc1.markdown(
                f"🔄 **교체율**<br>{turnover * 100:.0f}%",
                unsafe_allow_html=True
            )
            tc2.markdown(
                f"💰 **실제 거래비용**<br>{actual_tc * 100:.4f}%"
                f" ({actual_tc * 10000:.1f}bps)",
                unsafe_allow_html=True
            )
            tc3.markdown(
                f"🟢 **신규 매수** ({len(new_buys)}종목)<br>"
                + (", ".join(new_buys) if new_buys else "없음"),
                unsafe_allow_html=True
            )
            tc4.markdown(
                f"🔴 **매도** ({len(new_sells)}종목)<br>"
                + (", ".join(new_sells) if new_sells else "없음"),
                unsafe_allow_html=True
            )
            if kept:
                st.markdown(
                    f"⚪ **유지 종목** ({len(kept)}종목, 거래비용 없음): "
                    + ", ".join(f"**{t}**" for t in kept)
                )
            st.divider()

            data_types         = meta.get("data_types", {})
            sel_tickers_detail = (
                detail["selected_data"]["Ticker"].tolist()
                if "Ticker" in detail["selected_data"].columns else []
            )
            if data_types and sel_tickers_detail:
                tag_parts = []
                for t in sel_tickers_detail:
                    dtype = data_types.get(t, "없음")
                    icon  = "🟢" if dtype == "분기" else ("🟡" if dtype == "연간(분기 대체)" else "🔴")
                    tag_parts.append(f"{icon} **{t}** ({dtype})")
                st.markdown("**선정 종목 데이터 타입:**  " + "  ·  ".join(tag_parts))
                st.divider()

            d1, d2 = st.columns([3, 2])
            with d1:
                st.markdown("**선정 종목 상세**")
                st.dataframe(
                    detail["selected_data"].drop(
                        columns=["Target_Return", "Prediction"], errors="ignore"
                    ),
                    use_container_width=True, hide_index=True,
                )
            with d2:
                st.markdown("**지표 중요도 (Top 10)**")
                imp_df = detail["importance"].nlargest(10).reset_index()
                imp_df.columns = ["지표", "중요도"]
                fig_imp = px.bar(
                    imp_df, x="중요도", y="지표", orientation="h",
                    color="중요도", color_continuous_scale="Greens", height=300,
                )
                fig_imp.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False)
                st.plotly_chart(fig_imp, use_container_width=True, key=f"imp_{detail['date']}")

    # ── 지표 중요도 트렌드 ───────────────────────────────────
    st.subheader("🧠 지표 중요도 트렌드")
    imp_all_df = pd.DataFrame(importance_history).set_index("Date").fillna(0)
    top5       = imp_all_df.mean().nlargest(5).index.tolist()
    st.line_chart(imp_all_df[top5])

    imp_norm     = imp_all_df.div(imp_all_df.sum(axis=1), axis=0).fillna(0) * 100
    top7         = imp_norm.mean().nlargest(7).index.tolist()
    display_norm = imp_norm[top7].reset_index()
    fig_area = px.area(
        display_norm, x="Date", y=top7,
        title="주요 지표 영향력 비중 추이 (Top 7)",
        labels={"value": "중요도 비중 (%)", "Date": "리밸런싱 시점", "variable": "지표"},
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig_area.update_layout(
        hovermode="x unified", legend_orientation="h",
        legend_y=-0.2, yaxis_range=[0, 100],
    )
    st.plotly_chart(fig_area, use_container_width=True)
    display_importance_heatmap(imp_all_df)

    # ── 실시간 추천 ──────────────────────────────────────────
    st.divider()
    st.subheader(f"🎯 실시간 AI 추천 종목 (Next {reb_months}M)")
    latest_data = fetch_ml_data_optimized_pit(
        tickers, datetime.now(), full_hist_data, source_cache, is_training=False
    )
    recommend_all = pd.DataFrame()
    display_cols  = []
    if not latest_data.empty:
        X_latest = (
            latest_data.drop(["Ticker"], axis=1, errors="ignore")
            .reindex(columns=final_model_columns).fillna(0)
        )
        X_latest_scaled = pd.DataFrame(
            scaler.transform(X_latest), columns=final_model_columns
        )
        latest_data = latest_data.copy()
        latest_data["AI_Score"] = latest_model.predict(X_latest_scaled)
        recommend_all = latest_data.sort_values("AI_Score", ascending=False)
        display_cols  = ["Ticker", "AI_Score"] + [
            c for c in final_model_columns if c in recommend_all.columns
        ]
        st.dataframe(
            recommend_all[display_cols]
            .style.background_gradient(subset=["AI_Score"], cmap="YlGn")
            .format(precision=3),
            use_container_width=True, hide_index=True,
        )
    else:
        st.warning("실시간 추천 데이터를 생성할 수 없습니다.")

    # ── 다운로드 ─────────────────────────────────────────────
    st.divider()
    st.subheader("📥 분석 결과 내보내기")
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "📈 누적 수익률 (CSV)",
            data=cum_returns.to_csv().encode("utf-8-sig"),
            file_name=f"returns_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv", use_container_width=True,
        )
    with dl2:
        if not recommend_all.empty:
            st.download_button(
                "🎯 AI 추천 종목 (CSV)",
                data=recommend_all[display_cols].to_csv(index=False).encode("utf-8-sig"),
                file_name=f"recommendation_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv", use_container_width=True,
            )
    dl3, dl4 = st.columns(2)
    with dl3:
        if rebalance_details:
            history_dfs = []
            for detail in rebalance_details:
                tmp = detail["selected_data"].copy()
                tmp.insert(0, "Rebalance_Date", detail["date"])
                history_dfs.append(tmp)
            st.download_button(
                "📜 리밸런싱 히스토리 (CSV)",
                data=pd.concat(history_dfs, ignore_index=True).to_csv(index=False).encode("utf-8-sig"),
                file_name=f"history_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv", use_container_width=True,
            )
    with dl4:
        raw_list = []
        for detail in rebalance_details:
            raw_step = fetch_ml_data_optimized_pit(
                tickers, pd.to_datetime(detail["date"]),
                full_hist_data, source_cache, is_training=False,
            )
            raw_step.insert(0, "Data_Date", detail["date"])
            raw_list.append(raw_step)
        if raw_list:
            st.download_button(
                "📊 원본 피처 데이터 (CSV)",
                data=pd.concat(raw_list, ignore_index=True).to_csv(index=False).encode("utf-8-sig"),
                file_name=f"raw_features_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv", use_container_width=True,
            )

else:
    st.info("섹터를 선택하고 백테스트를 실행하세요.")