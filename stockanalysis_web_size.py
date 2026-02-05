import streamlit as st
import simfin as sf
from simfin.names import *
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. 환경 설정 ---
st.set_page_config(layout="wide", page_title="AI 퀀트 백테스트 분석기")

# 한글 폰트 설정
plt.rc('font', family='Malgun Gothic') 
plt.rc('axes', unicode_minus=False)

METRIC_DESC = {
    'P/E': '주가수익비율', 'P/S': '주가매출비율', 'P/B': '주가순자산비율', 'Forward_PE': '포워드 P/E', 'P_Cash': '주가현금비율',
    'PEG': '주가수익성장비율', 'PEG_Debt_Adj': '부채조정 PEG', 'FCF_Yield': '잉여현금흐름 수익률',
    'Sales_Growth': '매출성장률', 'EPS_Growth': 'EPS성장률', 'EPS_Growth_Next_Year': '내년 예상 EPS성장률',
    'Gross_Margin': '매출총이익률', 'Operating_Margin': '영업이익률', 'Profit_Margin': '순이익률',
    'ROA': '총자산수익률', 'ROE': '자기자본수익률', 'ROIC': '투하자본수익률', 'GP_A_Quality': 'GP/A', 'Sales_Growth_Proxy' : '과거 성장률(30%) + 내년 예상 성장률(40%) + 최근 주가 모멘텀(30%)',
    'Perf_Year': '1년 수익률', 'Perf_Half': '6개월 수익률', 'Perf_Month': '1개월 수익률', 'Perf_Week': '1주일 수익률', 'Performance_YTD': 'YTD 수익률',
    'Momentum_12M_1M': '12M-1M 모멘텀', 'Momentum_6M_1M': '6M-1M 모멘텀', 'Short_Term_Accel': '단기 가속도',
    'SMA_20_Rel': '20일 이격도', 'SMA_50_Rel': '50일 이격도', 'SMA_200_Rel': '200일 이격도', 'Beta': '베타',
    'RSI_Volatility_Adj': '변동성 조정 RSI', 'MA_Convergence': '이평선 수렴도',
    'Quick_Ratio': '당좌비율', 'LT_Debt_Equity': '장기부채비율', 'Total_Debt_Equity': '총부채비율',
    'Institutional_Transactions': '기관 수급 강도', 'Inst_Inside_Buy': '수급지수', 'Short_Squeeze_Potential': '숏스퀴즈 가능성'
}

sf.set_api_key('18ae7c59-5843-408e-8df9-314107ef4f2f')
sf.set_data_dir('simfin_data/')

# --- 2. 데이터 로드 및 전처리 ---
@st.cache_data(show_spinner="SimFin 데이터를 불러오는 중입니다...")
def load_and_process_data():
    df_inc = sf.load_income(variant='annual', market='us').reset_index()
    df_bal = sf.load_balance(variant='annual', market='us').reset_index()
    df_cf = sf.load_cashflow(variant='annual', market='us').reset_index()
    df_prices = sf.load_shareprices(variant='daily', market='us').reset_index()    
    df_prices[DATE] = pd.to_datetime(df_prices[DATE])
    df_prices = df_prices[df_prices[DATE] >= '2022-03-01'].reset_index(drop=True)
    # [핵심] CLOSE 외에 안 쓰는 컬럼(Open, High, Low 등)은 메모리에서 즉시 퇴출
    keep_cols = [TICKER, DATE, CLOSE]
    df_prices = df_prices[[c for c in keep_cols if c in df_prices.columns]]
    
    # [핵심] 숫자 정밀도 낮추기
    if CLOSE in df_prices.columns:
        df_prices[CLOSE] = df_prices[CLOSE].astype('float32')

    df_prices = df_prices.sort_values(by=[TICKER, DATE]).reset_index(drop=True)
    
    group = df_prices.groupby(TICKER)[CLOSE]
    df_prices['Perf_Year'] = group.pct_change(252)
    df_prices['Perf_Half'] = group.pct_change(126)
    df_prices['Perf_Month'] = group.pct_change(21)
    df_prices['Perf_Week'] = group.pct_change(5)
    df_prices['Vol_Month'] = group.transform(lambda x: x.pct_change().rolling(21).std())
    
    df_prices['SMA_20'] = group.transform(lambda x: x.rolling(20).mean())
    df_prices['SMA_50'] = group.transform(lambda x: x.rolling(50).mean())
    df_prices['SMA_200'] = group.transform(lambda x: x.rolling(200).mean())

    df_comp = sf.load_companies(market='us').reset_index()
    df_ind = sf.load_industries().reset_index()
    df_sector_map = pd.merge(df_comp[[TICKER, 'IndustryId']], df_ind, on='IndustryId')

    exclude = [TICKER, REPORT_DATE, 'SimFinId', 'Currency', 'Fiscal Year', 'Fiscal Period', 'Publish Date', 'Restated Date']
    list_inc = sorted([c for c in df_inc.columns if c not in exclude])
    list_bal = sorted([c for c in df_bal.columns if c not in exclude])
    list_cf = sorted([c for c in df_cf.columns if c not in exclude])
    list_sectors = sorted(df_sector_map['Industry'].unique().tolist())

    return df_inc, df_bal, df_cf, df_prices, df_sector_map, list_inc, list_bal, list_cf, list_sectors

df_inc, df_bal, df_cf, df_prices, df_sector_map, list_inc, list_bal, list_cf, list_sectors = load_and_process_data()

# --- 3. 사이드바 UI ---
st.sidebar.title("🛠️ 분석 설정")
cap_options = ['Any', 'Mega ($200bln+)', '+Large (over $10bln)', '+Mid (over $2bln)', '+Small (over $300mln)']
cap_choice = st.sidebar.selectbox("Market Cap 범위", cap_options, index=2)
top_n = st.sidebar.number_input("투자 종목 수", min_value=1, max_value=50, value=3)

rebalance_options = {'Monthly (1M)': 'ME', 'Quarterly (3M)': 'QE', 'Semi-Annually (6M)': '6ME',  'Annually (12M)': 'YE'}
reb_choice_label = st.sidebar.selectbox("리밸런싱 주기", list(rebalance_options.keys()), index=1)
reb_freq = rebalance_options[reb_choice_label]

start_date = st.sidebar.date_input("백테스트 시작일", df_prices[DATE].min().date() + timedelta(days=0))
end_date = st.sidebar.date_input("백테스트 종료일", df_prices[DATE].max().date())

all_features = sorted(list(set(list_inc + list_bal + list_cf + list(METRIC_DESC.keys()))))
selected_sectors = st.sidebar.multiselect("섹터 선택", list_sectors, default=['Computer Hardware', 'Semiconductors'])
selected_features = st.sidebar.multiselect("지표 선택", all_features, default=['P/E', 'P/S', 'P/B', 'Forward_PE', 'P_Cash', 'PEG', 'PEG_Debt_Adj', 'FCF_Yield', 
'Sales_Growth', 'EPS_Growth', 'Gross_Margin', 'Operating_Margin', 
'Profit_Margin', 'ROA', 'ROE', 'ROIC', 'GP_A_Quality', 'Sales_Growth_Proxy', 
'Perf_Year', 'Perf_Half', 'Perf_Month', 'Perf_Week', 'Performance_YTD', 
'Momentum_12M_1M', 'Momentum_6M_1M', 'Short_Term_Accel', 
'SMA_20_Rel', 'SMA_50_Rel', 'SMA_200_Rel', 'Beta', 'RSI_Volatility_Adj', 
'MA_Convergence', 'Quick_Ratio', 'LT_Debt_Equity', 'Total_Debt_Equity', 
'Institutional_Transactions', 'Inst_Inside_Buy', 'Short_Squeeze_Potential'])

# --- 4. 분석 실행 ---
if st.sidebar.button("🚀 퀀트 분석 시작"):
    if not selected_sectors or not selected_features:
        st.error("섹터와 지표를 선택해주세요.")
    elif start_date >= end_date:
        st.error("시작일은 종료일보다 빨라야 합니다.")
    else:
        with st.spinner("머신러닝 학습 및 백테스트 진행 중..."):
            df_i, df_b, df_c = df_inc.copy(), df_bal.copy(), df_cf.copy()
            df_p = df_prices.copy()
            
            df_i[REPORT_DATE] = pd.to_datetime(df_i[REPORT_DATE])
            df_b[REPORT_DATE] = pd.to_datetime(df_b[REPORT_DATE])
            df_c[REPORT_DATE] = pd.to_datetime(df_c[REPORT_DATE])
            df_p[DATE] = pd.to_datetime(df_p[DATE])

            df = pd.merge(df_i, df_b, on=[TICKER, REPORT_DATE], suffixes=('', '_bal'))
            df = pd.merge(df, df_c, on=[TICKER, REPORT_DATE], suffixes=('', '_cf'))
            df = pd.merge(df, df_sector_map[[TICKER, 'Industry']], on=TICKER)
            df = df[df['Industry'].isin(selected_sectors)]
            
            df_p = df_p.sort_values([TICKER, DATE])
            df_p_indexed = df_p.set_index(DATE)

            def calc_next_ret(group):
                resampled = group[CLOSE].resample(reb_freq).last()
                return resampled.pct_change(fill_method=None).shift(-1)

            df_ann_ret = df_p_indexed.groupby(TICKER).apply(calc_next_ret).reset_index()
            df_ann_ret.columns = [TICKER, REPORT_DATE, 'Next_Return']
            df_ann_ret[REPORT_DATE] = pd.to_datetime(df_ann_ret[REPORT_DATE])

            train_start = pd.Timestamp(start_date) - timedelta(days=365)
            df = df[(df[REPORT_DATE] >= train_start) & (df[REPORT_DATE] <= pd.Timestamp(end_date))]

            # [수정 포인트 1] 날짜 정규화 및 병합 전 중복 제거
            df = df.sort_values([TICKER, REPORT_DATE]).drop_duplicates(subset=[TICKER, REPORT_DATE])
            
            df_ml = pd.merge_asof(
                df.sort_values(REPORT_DATE),
                df_p.sort_values(DATE),
                by=TICKER,
                left_on=REPORT_DATE,
                right_on=DATE,
                direction='backward'
            )
            
            # [수정 포인트 2] 수익률 병합 시 날짜 오차 허용 범위 설정 (시차 문제 해결)
            df_ml = pd.merge_asof(
                df_ml.sort_values(REPORT_DATE),
                df_ann_ret.dropna().sort_values(REPORT_DATE),
                by=TICKER,
                on=REPORT_DATE,
                direction='nearest',
                tolerance=pd.Timedelta(days=7) # 7일 이내의 날짜 차이는 동일 리밸런싱 시점으로 간주
            )

            df_ml['Market_Cap'] = df_ml[CLOSE] * df_ml['Shares (Diluted)']
            m = df_ml['Market_Cap']
            if cap_choice == 'Mega ($200bln+)': df_ml = df_ml[m >= 200e9]
            elif cap_choice == '+Large (over $10bln)': df_ml = df_ml[m >= 10e9]
            elif cap_choice == '+Mid (over $2bln)': df_ml = df_ml[m >= 2e9]
            elif cap_choice == '+Small (over $300mln)': df_ml = df_ml[m >= 300e6]

            if df_ml.empty:
                st.error("조건에 맞는 데이터가 없습니다.")
                st.stop()
                
            df_ml = df_ml.sort_values([TICKER, REPORT_DATE])
            m_safe = df_ml['Market_Cap'].replace(0, np.nan)
            df_ml['P/E'] = m_safe / df_ml['Net Income'].replace(0, np.nan)
            df_ml['P/S'] = m_safe / df_ml['Revenue'].replace(0, np.nan)
            df_ml['P/B'] = m_safe / df_ml['Total Equity'].replace(0, np.nan)
            df_ml['P_Cash'] = m_safe / df_ml['Cash, Cash Equivalents & Short Term Investments'].replace(0, np.nan)
            df_ml['Gross_Margin'] = df_ml['Gross Profit'] / df_ml['Revenue'].replace(0, np.nan)
            df_ml['Operating_Margin'] = df_ml['Operating Income (Loss)'] / df_ml['Revenue'].replace(0, np.nan)
            df_ml['Profit_Margin'] = df_ml['Net Income'] / df_ml['Revenue'].replace(0, np.nan)
            df_ml['GP_A_Quality'] = df_ml['Gross Profit'] / df_ml['Total Assets'].replace(0, np.nan)
            df_ml['ROE'] = df_ml['Net Income'] / df_ml['Total Equity'].replace(0, np.nan)
            df_ml['ROA'] = df_ml['Net Income'] / df_ml['Total Assets'].replace(0, np.nan)
            df_ml['ROIC'] = df_ml['Operating Income (Loss)'] / (df_ml['Total Assets'] - df_ml['Total Current Liabilities']).replace(0, np.nan).abs()
            fcf = df_ml['Net Cash from Operating Activities'].fillna(0) + df_ml['Change in Fixed Assets & Intangibles'].fillna(0)
            df_ml['FCF_Yield'] = fcf / m_safe
            df_ml['Sales_Growth'] = df_ml.groupby(TICKER)['Revenue'].pct_change()
            df_ml['EPS_Growth'] = df_ml.groupby(TICKER)['Net Income'].pct_change()
            
            # [수정 포인트 3] 미래 참조 지표(Next_Year) 제거 로직 반영 (사용자 요청 시 유지하되 계산 방식 주의)
            # 여기서는 백테스트 무결성을 위해 기존 EPS_Growth_Next_Year 대신 과거 성장률 추세로 대체 권장하나 
            # 사용자 질문의 맥락을 고려해 코드는 유지하되 '미래 데이터'임을 명시합니다.
            if 'EPS_Growth_Next_Year' in selected_features:
                 df_ml['EPS_Growth_Next_Year'] = df_ml.groupby(TICKER)['Net Income'].pct_change().shift(-1)

            df_ml['Total_Debt_Equity'] = df_ml['Total Liabilities'] / df_ml['Total Equity'].replace(0, np.nan)
            df_ml['LT_Debt_Equity'] = df_ml['Long Term Debt'] / df_ml['Total Equity'].replace(0, np.nan)
            eps_g_pct = df_ml['EPS_Growth'] * 100
            df_ml['PEG'] = df_ml['P/E'] / eps_g_pct.apply(lambda x: x if x > 0 else np.nan)
            df_ml['PEG_Debt_Adj'] = df_ml['PEG'] * (df_ml['Total_Debt_Equity'] + 1)
            df_ml['Estimated_Fwd_GP'] = df_ml['Gross Profit'] * (1 + df_ml['EPS_Growth'].fillna(0))
            df_ml['Short_Squeeze_Potential'] = df_ml['Total_Debt_Equity'] * df_ml['Vol_Month']
            df_ml['MA_Convergence'] = df_ml['SMA_20'] / df_ml['SMA_50'].replace(0, np.nan)
            df_ml['Short_Term_Accel'] = df_ml['Perf_Month'] - df_ml['Perf_Week']
            df_ml['SMA_20_Rel'] = df_ml[CLOSE] / df_ml['SMA_20'].replace(0, np.nan)
            df_ml['SMA_50_Rel'] = df_ml[CLOSE] / df_ml['SMA_50'].replace(0, np.nan)
            df_ml['SMA_200_Rel'] = df_ml[CLOSE] / df_ml['SMA_200'].replace(0, np.nan)
            df_ml['Year_Start_Price'] = df_ml.groupby([TICKER, df_ml[REPORT_DATE].dt.year])[CLOSE].transform('first')
            df_ml['Performance_YTD'] = (df_ml[CLOSE] - df_ml['Year_Start_Price']) / df_ml['Year_Start_Price']
            df_ml['Institutional_Transactions'] = df_ml.groupby(TICKER)['Shares (Diluted)'].pct_change()
            df_ml['Inst_Inside_Buy'] = df_ml['Institutional_Transactions'].fillna(0)
            
            main_perf = df_ml['Perf_Year'].replace(0, np.nan).combine_first(df_ml['Perf_Half'].replace(0, np.nan))
            df_ml['Momentum_12M_1M'] = df_ml['Perf_Year'] - df_ml['Perf_Month']
            df_ml['Momentum_6M_1M'] = df_ml['Perf_Half'] - df_ml['Perf_Month']

            def cal_rsi(s, n=14):
                diff = s.diff()
                up = diff.clip(lower=0).rolling(n).mean()
                down = -diff.clip(upper=0).rolling(n).mean()
                return 100 - (100 / (1 + (up / down.replace(0, np.nan))))
            df_ml['RSI_14'] = df_ml.groupby(TICKER)[CLOSE].transform(cal_rsi)
            df_ml['RSI_Volatility_Adj'] = df_ml['RSI_14'] / (df_ml['Vol_Month'] + 0.1)

            # Sales_Growth_Proxy 계산 시 EPS_Growth_Next_Year 참조 에러 방지
            next_growth = df_ml['EPS_Growth_Next_Year'] if 'EPS_Growth_Next_Year' in df_ml.columns else df_ml['EPS_Growth']
            df_ml['Sales_Growth_Proxy'] = (df_ml['Sales_Growth'].fillna(0) * 0.3) + \
                                         (next_growth.fillna(0) * 0.4) + \
                                         ((df_ml['Perf_Month']+df_ml['Perf_Week'])/2).fillna(0) * 0.3
            
            fwd_earnings_proxy = df_ml['Net Income'] * (1 + df_ml['EPS_Growth'].fillna(0))
            df_ml['Forward_PE'] = m_safe / fwd_earnings_proxy.replace(0, np.nan)
            df_ml['Forward_PE'] = df_ml['Forward_PE'].fillna(df_ml['P/E'])

            target_col = 'Next_Return'
            existing_features = [col for col in selected_features if col in df_ml.columns]

            for col in existing_features:
                df_ml[col] = df_ml.groupby('Industry')[col].transform(lambda x: x.fillna(x.median()))
                df_ml[col] = df_ml[col].fillna(df_ml[col].median())
            
            for col in existing_features:
                df_ml[col] = df_ml.groupby('Industry')[col].rank(pct=True)
            
            df_ml[existing_features] = df_ml[existing_features].fillna(0.5)
            df_ml = df_ml.dropna(subset=[target_col])
            df_ml[existing_features] = df_ml[existing_features].replace([np.inf, -np.inf], 0.5)

            # [수정 포인트 4] 최종 리밸런싱 날짜 정규화 및 중복 제거 (횟수 정상화의 핵심)
            # 날짜를 해당 분기의 마지막 날로 통일하여 '파편화된 날짜'를 하나로 합칩니다.
            df_ml[REPORT_DATE] = pd.to_datetime(df_ml[REPORT_DATE]).dt.to_period(reb_freq[0]).dt.to_timestamp()
            df_ml = df_ml.sort_values([TICKER, REPORT_DATE]).drop_duplicates(subset=[TICKER, REPORT_DATE], keep='last')

            st.subheader("🔍 데이터 무결성 검사")
            st.write(f"✅ 최종 유효 리밸런싱 시점 수: {len(df_ml[REPORT_DATE].unique())}회")
            data_counts = df_ml[existing_features + [target_col]].count().reset_index()
            st.dataframe(data_counts)

            # --- 5단계: 모델 학습 및 예측 ---
            if len(df_ml) > 10:
                X = df_ml[existing_features]
                y = df_ml[target_col]
                if isinstance(y, pd.DataFrame): y = y.iloc[:, 0]

                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X, y)
                preds = model.predict(X)
                df_ml['Pred'] = preds[:, 0] if preds.ndim > 1 else preds
            else:
                st.error("데이터가 부족합니다.")
                st.stop()

            # --- 6단계: 백테스트 시뮬레이션 ---
            daily_returns = []
            dates = sorted(df_ml[REPORT_DATE].unique())
            
            for i in range(len(dates)-1):
                cur_date, nxt_date = dates[i], dates[i+1]
                current_pool = df_ml[df_ml[REPORT_DATE] == cur_date]
                if current_pool.empty: continue
                
                top_tickers = current_pool.nlargest(top_n, 'Pred')[TICKER].tolist()
                # 공시 시차 고려: 리밸런싱 기준일 + 2일 뒤부터 매수 시작
                trade_start = cur_date + pd.Timedelta(days=2)
                
                period_p = df_prices[(df_prices[TICKER].isin(top_tickers)) & (df_prices[DATE] >= trade_start) & (df_prices[DATE] <= nxt_date)]
                
                if not period_p.empty:
                    daily_pct = period_p.pivot(index=DATE, columns=TICKER, values=CLOSE).pct_change().mean(axis=1)
                    daily_returns.append(daily_pct.dropna())

            if daily_returns:
                df_daily_res = pd.concat(daily_returns)
                df_daily_res = df_daily_res[~df_daily_res.index.duplicated(keep='first')]
                df_cumulative = (1 + df_daily_res).cumprod()

                st.success("분석 완료!")
                col1, col2, col3 = st.columns(3)
                col1.metric("누적 수익률", f"{(df_cumulative.iloc[-1]-1)*100:.2f}%")
                col2.metric("MDD", f"{((df_cumulative - df_cumulative.cummax())/df_cumulative.cummax()).min()*100:.2f}%")
                col3.metric("리밸런싱 횟수", f"{len(dates)}회")

                tab1, tab2 = st.tabs(["📈 수익률 추이", "📊 중요 지표"])
                with tab1:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_cumulative.index, y=df_cumulative.values, name="전략"))
                    st.plotly_chart(fig, use_container_width=True)
                with tab2:
                    # 1. 데이터 준비: 중요도 계산 및 누적 합계 추가
                    feat_imp = pd.DataFrame({'지표명': existing_features, '중요도': model.feature_importances_}).sort_values('중요도', ascending=False)
                    feat_imp['누적 중요도'] = feat_imp['중요도'].cumsum()
                    
                    # 2. 파레토 차트 그리기 (막대 + 라인)
                    fig_pareto = go.Figure()

                    # 막대 그래프 (개별 중요도)
                    fig_pareto.add_trace(go.Bar(x=feat_imp['지표명'], y=feat_imp['중요도'], name='개별 중요도', marker_color='rgb(55, 83, 109)'))

                    # 선 그래프 (누적 중요도)
                    fig_pareto.add_trace(go.Scatter(x=feat_imp['지표명'], y=feat_imp['누적 중요도'], name='누적 중요도', yaxis='y2', line=dict(color='rgb(219, 64, 82)', width=3)))

                    # 레이아웃 설정 (이중 축 적용)
                    fig_pareto.update_layout(
                        title='지표 중요도 파레토 분석 (Feature Importance Pareto)',
                        xaxis=dict(title='Financial Metrics'),
                        yaxis=dict(title='개별 중요도', showgrid=True),
                        yaxis2=dict(
                            title='누적 중요도', overlaying='y', side='right', range=[0, 1.05], tickformat='.0%', showgrid=False),
                        legend=dict(x=0.8, y=1.2, orientation='h'),
                        margin=dict(l=50, r=50, t=80, b=50),
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_pareto, use_container_width=True)

                    # 3. 상세 중요도 테이블 추가
                    st.markdown("---")
                    st.subheader("📊 지표별 상세 중요도 테이블")
                    
                    # 표시용 데이터프레임 포맷팅
                    display_df = feat_imp.copy()
                    display_df['중요도 비중'] = display_df['중요도'].apply(lambda x: f"{x:.2%}")
                    display_df['누적 비중'] = display_df['누적 중요도'].apply(lambda x: f"{x:.2%}")
                    
                    # 테이블 출력
                    st.table(display_df[['지표명', '중요도 비중', '누적 비중']])
            else:
                st.warning("조건에 맞는 결과가 없습니다.")