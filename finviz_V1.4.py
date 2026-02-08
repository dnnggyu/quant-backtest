import streamlit as st
import pandas as pd
import numpy as np
from finvizfinance.screener.overview import Overview
from finvizfinance.screener.valuation import Valuation
from finvizfinance.screener.financial import Financial
from finvizfinance.screener.ownership import Ownership
from finvizfinance.screener.performance import Performance
from finvizfinance.screener.technical import Technical
import yfinance as yf
import plotly.express as px



st.set_page_config(layout="wide")

# 상단 공백 제거를 위한 CSS
st.markdown("""
    <style>
    /* 여백 설정 유지 */
    .stMainBlockContainer {
        padding-top: 1.5rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }

    /* 아이콘 및 툴바 정밀 타격 */
    /* 1. 하단 툴바 전체 (종이배 아이콘 포함) */
    [data-testid="stToolbar"], .stToolbar {
        display: none !important;
        height: 0px !important;
        width: 0px !important;
    }

    /* 2. 상태 위젯 (분홍색 아이콘 포함) */
    [data-testid="stStatusWidget"], .stStatusWidget {
        display: none !important;
        visibility: hidden !important;
    }

    /* 3. 모바일 하단 플로팅 요소 강제 숨김 */
    div[class*="st-emotion-cache-1ky89f3"], 
    div[class*="st-emotion-cache-18ni7ap"] {
        display: none !important;
    }

    /* 4. 기타 헤더/푸터 */
    header, footer, #MainMenu { display: none !important; }
    
    @media (max-width: 480px) {
        .stMainBlockContainer { padding-top: 0rem !important; }
        [data-testid="stImage"] { margin-top: -1rem !important; }
    }
    </style>
    """, unsafe_allow_html=True)


# 2. 우측 상단 메뉴와 GitHub 아이콘을 숨기는 CSS
st.markdown("""
    <style>
    /* 상단 헤더 전체 숨기기 */
    header {visibility: hidden !important;}
    /* 메뉴 버튼 숨기기 */
    #MainMenu {visibility: hidden !important;}
    /* 배포 버튼(GitHub 아이콘 포함) 숨기기 */
    .stAppDeployButton {display:none !important;}
    /* 하단 푸터 숨기기 */
    footer {visibility: hidden !important;}
    </style>
""", unsafe_allow_html=True)

# --- 1. 페이지 설정 및 상수 정의 ---
st.set_page_config(page_title="Stock Strategy Analyzer", layout="wide")
st.image("banner.png", use_container_width=True)
#st.markdown("## 📊 Quant 투자 종목 선정")

# Finviz 섹터-산업 매핑 데이터
SECTOR_INDUSTRY_MAP = {
    "Basic Materials": ["Agricultural Inputs", "Aluminum", "Chemicals", "Copper", "Gold", "Lumber & Wood Production", "Other Precious Metals & Mining", "Other Industrial Metals & Mining", "Paper & Paper Products", "Specialty Chemicals", "Steel"],
    "Communication Services": ["Advertising Agencies", "Broadcasting", "Entertainment", "Internet Content & Information", "Publishing", "Telecom Services"],
    "Consumer Cyclical": ["Apparel Manufacturing", "Apparel Retail", "Auto & Truck Dealerships", "Auto Manufacturers", "Auto Parts", "Footwear & Accessories", "Gambling", "Home Improvement Retail", "Hotels, Motels & Resorts", "Internet Retail", "Luxury Goods", "Packaging & Containers", "Personal Services", "Residential Construction", "Resorts & Casinos", "Restaurants", "Specialty Retail", "Textile Manufacturing", "Travel Services"],
    "Consumer Defensive": ["Beverages - Non-Alcoholic", "Beverages - Wineries & Distilleries", "Confectioners", "Discount Stores", "Education & Training Services", "Farm Products", "Food Distribution", "Grocery Stores", "Household & Personal Products", "Packaged Foods", "Tobacco"],
    "Energy": ["Oil & Gas Drilling", "Oil & Gas E&P", "Oil & Gas Equipment & Services", "Oil & Gas Integrated", "Oil & Gas Midstream", "Oil & Gas Refining & Marketing", "Solar", "Thermal Energy", "Uranium"],
    "Financial": ["Asset Management", "Banks - Diversified", "Banks - Regional", "Capital Markets", "Credit Services", "Financial Data & Stock Exchanges", "Insurance - Diversified", "Insurance - Life", "Insurance - Property & Casualty", "Insurance - Reinsurance", "Insurance - Specialty", "Insurance Brokers", "Mortgage Finance", "Shell Companies"],
    "Healthcare": ["Biotechnology", "Diagnostics & Research", "Drug Manufacturers - General", "Drug Manufacturers - Specialty & Generic", "Health Information Services", "Healthcare Plans", "Medical Care Facilities", "Medical Devices", "Medical Instruments & Supplies", "Pharmaceutical Retailers"],
    "Industrials": ["Aerospace & Defense", "Airlines", "Building Products & Equipment", "Business Equipment & Services", "Conglomerates", "Consulting Services", "Electrical Equipment & Parts", "Engineering & Construction", "Farm & Heavy Construction Machinery", "Industrial Distribution", "Integrated Freight & Logistics", "Marine Shipping", "Metal Fabrication", "Pollution & Treatment Controls", "Railroads", "Rental & Leasing Services", "Security & Protection Services", "Specialty Business Services", "Staffing & Employment Services", "Tools & Accessories", "Trucking"],
    "Real Estate": ["REIT - Diversified", "REIT - Healthcare Facilities", "REIT - Hotel & Motel", "REIT - Industrial", "REIT - Mortgage", "REIT - Office", "REIT - Residential", "REIT - Retail", "REIT - Specialty", "Real Estate - Development", "Real Estate Services"],
    "Technology": ["Communication Equipment", "Computer Hardware", "Consumer Electronics", "Electronic Components", "Electronics & Computer Distribution", "Information Technology Services", "Scientific & Technical Instruments", "Semiconductor Equipment & Materials", "Semiconductors", "Software - Application", "Software - Infrastructure", "Solar"],
    "Utilities": ["Utilities - Diversified", "Utilities - Independent Power Producers", "Utilities - Regulated Electric", "Utilities - Regulated Gas", "Utilities - Regulated Water"]
}

FINVIZ_COL_MAP = {
    # --- [기존 항목 유지] ---
    'Perf Year': 'Performance (Year)', 
    'Perf Half': 'Performance (Half Year)',
    'Perf Month': 'Performance (Month)', 
    'Perf Week': 'Performance (Week)',
    'Gross M': 'Gross Margin', 
    'ROA': 'Return on Assets', 
    'ROE': 'Return on Equity',
    'P/FCF': 'P/Free Cash Flow', 
    'Inst Trans': 'Institutional Transactions',
    'Insider Trans': 'Insider Transactions', 
    'Short Float': 'Short Float',
    'Rel Volume': 'Relative Volume', 
    'Fwd P/E': 'Forward P/E', 
    'P/B': 'P/B', 
    'P/S': 'P/S',
    'EPS Next Y': 'EPS Growth Next Year', 
    'EPS This Y': 'EPS Growth This Year',
    'SMA20': '20-Day Simple Moving Average', 
    'SMA50': '50-Day Simple Moving Average',
    'SMA200': '200-Day Simple Moving Average', 
    'RSI': 'Relative Strength Index (14)',
    'Volatility M': 'Volatility (Month)', 
    'Volatility W': 'Volatility (Week)',
    'Sales Past 5Y': 'Sales Growth Past 5 Years', 
    'Curr R': 'Current Ratio',

    # --- [누락 방지를 위해 추가된 항목] ---
    'Perf YTD': 'Performance (YTD)',           # config의 'Performance (YTD)' 대응
    'Perf Quart': 'Performance (Quarter)',     # config의 'Performance (Quarter)' 대응
    'PEG': 'PEG',                              # 스코어링 및 config용
    'Oper M': 'Oper Margin',                   # 전처리 % 연산용
    'Profit M': 'Profit Margin',               # 전처리 % 연산용
    'P/E': 'P/E',                              # 기본 밸류에이션
    'P/C': 'P/Cash',                           # 가치 지표 보강
    'Dividend': 'Dividend Yield',              # 배당 수익률
    'Debt/Eq': 'Total Debt/Equity',            # 부채 비율 (안정성)
    '52W High': '52W High',                    # 기술적 지표 (신고가 근접도)
    '52W Low': '52W Low'                       # 기술적 지표 (신저가 근접도)
}

# 섹터별 가중치 (작성하신 전략 그대로 반영)
STRATEGIES = {
    "Technology": {"Momentum_Custom": 25, "GP_A_Quality": 15, "Estimated_Fwd_GP": 15, "Sales_Growth_Proxy": 15, "FCF_Yield": 15, "PEG_Debt_Adj": 15},
    "Communication Services": {"Momentum_Custom": 25, "Short_Squeeze_Potential": 25, "Inst_Inside_Buy": 20, "Forward P/E": 15, "Risk_Adj_Return": 15},
    "Consumer Cyclical": {"Risk_Adj_Return": 25, "Momentum_6M_1M": 20, "P/S": 20, "GPA_Score_Internal": 20, "RSI_Volatility_Adj": 15},
    "Financial": {"Return on Equity": 30, "P/B": 30, "Dividend Yield": 20, "Net_Working_Capital_Value": 10, "Institutional Transactions": 10},
    "Healthcare": {"GPA_Score_Internal": 25, "Risk_Adj_Return": 25, "EPS Growth Next 5 Years": 20, "P/Cash": 15, "Current Ratio": 15},
    "Consumer Defensive": {"Dividend Yield": 30, "FCF_Yield": 20, "GPA_Score_Internal": 20, "Volatility (Month)": 15, "Total Debt/Equity": 15},
    "Energy": {"FCF_Yield": 30, "Momentum_12M_1M": 20, "Return on Invested Capital": 20, "Risk_Adj_Return": 15, "Dividend Yield": 15},
    "Industrials": {"Operating Margin": 25, "Sales_Growth_Proxy": 20, "Forward P/E": 20, "Net_Working_Capital_Value": 20, "Total Debt/Equity": 15},
    "Basic Materials": {"Momentum_6M_1M": 25, "FCF_Yield": 25, "Risk_Adj_Return": 20, "Return on Assets": 15, "P/E": 15},
    "Utilities": {"Dividend Yield": 35, "Total Debt/Equity": 25, "FCF_Yield": 20, "P/E": 10, "Risk_Adj_Return": 10},
    "Real Estate": {"FCF_Yield": 30, "Dividend Yield": 30, "Inst_Inside_Buy": 20, "Net_Working_Capital_Value": 10, "Total Debt/Equity": 10}
}

DISPLAY_MAP = {
    "TotalRevenue": "Total Revenue",
    "OperatingRevenue": "Operating Revenue",
    "CostOfRevenue": "Cost of Revenue",
    "ReconciledCostOfRevenue": "Reconciled Cost of Revenue",
    "GrossProfit": "Gross Profit",
    "OperatingExpense": "Operating Expense",
    "SellingGeneralAndAdministration": "Selling General and Administration",
    "ResearchAndDevelopment": "Research and Development",
    "OtherOperatingExpenses": "Other Operating Expenses",
    "OperatingIncome": "Operating Income",
    "TotalOperatingIncomeAsReported": "Total Operating Income as Reported",
    "TotalExpenses": "Total Expenses",
    "NetNonOperatingInterestIncomeExpense": "Net Non Operating Interest Income Expense",
    "InterestIncomeNonOperating": "Interest Income Non Operating",
    "InterestExpenseNonOperating": "Interest Expense Non Operating",
    "NetInterestIncome": "Net Interest Income",
    "InterestIncome": "Interest Income",
    "InterestExpense": "Interest Expense",
    "OtherIncomeExpense": "Other Income Expense",
    "OtherNonOperatingIncomeExpenses": "Other Non Operating Income Expenses",
    "EarningsFromEquityInterestNetOfTax": "Earnings from Equity Interest Net of Tax",
    "PretaxIncome": "Pretax Income",
    "TotalUnusualItems": "Total Unusual Items",
    "TotalUnusualItemsExcludingGoodwill": "Total Unusual Items Excluding Goodwill",
    "SpecialIncomeCharges": "Special Income Charges",
    "GainOnSaleOfPPE": "Gain on Sale of PPE",
    "GainOnSaleOfSecurity": "Gain on Sale of Security",
    "OtherSpecialCharges": "Other Special Charges",
    "WriteOff": "Write Off",
    "ImpairmentOfCapitalAssets": "Impairment of Capital Assets",
    "RestructuringAndMergernAcquisition": "Restructuring and Merger n Acquisition",
    "TaxProvision": "Tax Provision",
    "NetIncomeContinuousOperations": "Net Income Continuous Operations",
    "NetIncomeIncludingNoncontrollingInterests": "Net Income Including Noncontrolling Interests",
    "MinorityInterests": "Minority Interests",
    "NetIncomeFromContinuingOperationNetMinorityInterest": "Net Income from Continuing Operation Net Minority Interest",
    "NetIncomeFromContinuingAndDiscontinuedOperation": "Net Income from Continuing and Discontinued Operation",
    "NetIncome": "Net Income",
    "NetIncomeCommonStockholders": "Net Income Common Stockholders",
    "NormalizedIncome": "Normalized Income",
    "BasicEPS": "Basic EPS",
    "DilutedEPS": "Diluted EPS",
    "BasicAverageShares": "Basic Average Shares",
    "DilutedAverageShares": "Diluted Average Shares",
    "AverageDilutionEarnings": "Average Dilution Earnings",
    "DilutedNIAvailtoComStockholders": "Diluted NI Avail to Com Stockholders",
    "EBIT": "EBIT",
    "EBITDA": "EBITDA",
    "NormalizedEBITDA": "Normalized EBITDA",
    "ReconciledDepreciation": "Reconciled Depreciation",
    "TaxRateForCalcs": "Tax Rate for Calcs",
    "TaxEffectOfUnusualItems": "Tax Effect of Unusual Items",

    # --- [Assets: Current Assets (유동자산)] ---
    "CashCashEquivalentsAndShortTermInvestments": "Cash, Cash Equivalents & Short Term Investments",
    "CashAndCashEquivalents": "Cash and Cash Equivalents",
    "CashFinancial": "Cash Financial",
    "CashEquivalents": "Cash Equivalents",
    "OtherShortTermInvestments": "Other Short Term Investments",
    "Receivables": "Receivables",
    "AccountsReceivable": "Accounts Receivable",
    "TaxesReceivable": "Taxes Receivable",
    "OtherReceivables": "Other Receivables",
    "Inventory": "Inventory",
    "RawMaterials": "Raw Materials",
    "WorkInProcess": "Work In Process",
    "FinishedGoods": "Finished Goods",
    "AssetsHeldForSaleCurrent": "Assets Held For Sale Current",
    "OtherCurrentAssets": "Other Current Assets",
    "CurrentAssets": "Total Current Assets",

    # --- [Assets: Non-Current Assets (비유동자산)] ---
    "NetPPE": "Net Property, Plant and Equipment",
    "GrossPPE": "Gross Property, Plant and Equipment",
    "Properties": "Properties",
    "LandAndImprovements": "Land and Improvements",
    "BuildingsAndImprovements": "Buildings and Improvements",
    "MachineryFurnitureEquipment": "Machinery, Furniture & Equipment",
    "OtherProperties": "Other Properties",
    "ConstructionInProgress": "Construction In Progress",
    "AccumulatedDepreciation": "Accumulated Depreciation",
    "GoodwillAndOtherIntangibleAssets": "Goodwill and Other Intangible Assets",
    "Goodwill": "Goodwill",
    "OtherIntangibleAssets": "Other Intangible Assets",
    "InvestmentsAndAdvances": "Investments and Advances",
    "InvestmentinFinancialAssets": "Investment in Financial Assets",
    "AvailableForSaleSecurities": "Available For Sale Securities",
    "NonCurrentDeferredAssets": "Non Current Deferred Assets",
    "NonCurrentDeferredTaxesAssets": "Non Current Deferred Taxes Assets",
    "OtherNonCurrentAssets": "Other Non Current Assets",
    "TotalNonCurrentAssets": "Total Non Current Assets",
    "TotalAssets": "Total Assets",

    # --- [Liabilities: Current Liabilities (유동부채)] ---
    "PayablesAndAccruedExpenses": "Payables and Accrued Expenses",
    "Payables": "Payables",
    "AccountsPayable": "Accounts Payable",
    "TotalTaxPayable": "Total Tax Payable",
    "OtherPayable": "Other Payable",
    "CurrentAccruedExpenses": "Current Accrued Expenses",
    "CurrentDebtAndCapitalLeaseObligation": "Current Debt and Capital Lease Obligation",
    "CurrentDebt": "Current Debt",
    "OtherCurrentBorrowings": "Other Current Borrowings",
    "CurrentNotesPayable": "Current Notes Payable",
    "CurrentCapitalLeaseObligation": "Current Capital Lease Obligation",
    "OtherCurrentLiabilities": "Other Current Liabilities",
    "CurrentLiabilities": "Total Current Liabilities",

    # --- [Liabilities: Non-Current Liabilities (비유동부채)] ---
    "LongTermDebtAndCapitalLeaseObligation": "Long Term Debt and Capital Lease Obligation",
    "LongTermDebt": "Long Term Debt",
    "LongTermCapitalLeaseObligation": "Long Term Capital Lease Obligation",
    "NonCurrentDeferredLiabilities": "Non Current Deferred Liabilities",
    "NonCurrentDeferredRevenue": "Non Current Deferred Revenue",
    "OtherNonCurrentLiabilities": "Other Non Current Liabilities",
    "TotalNonCurrentLiabilitiesNetMinorityInterest": "Total Non Current Liabilities",
    "TotalLiabilitiesNetMinorityInterest": "Total Liabilities",

    # --- [Equity: Stockholders' Equity (자본)] ---
    "StockholdersEquity": "Stockholders' Equity",
    "TotalEquityGrossMinorityInterest": "Total Equity Gross Minority Interest",
    "CommonStockEquity": "Common Stock Equity",
    "CapitalStock": "Capital Stock",
    "CommonStock": "Common Stock",
    "AdditionalPaidInCapital": "Additional Paid In Capital",
    "RetainedEarnings": "Retained Earnings",
    "TreasuryStock": "Treasury Stock",
    "GainsLossesNotAffectingRetainedEarnings": "Gains Losses Not Affecting Retained Earnings",
    "OtherEquityAdjustments": "Other Equity Adjustments",
    
    # --- [Other Metrics & Supplemental] ---
    "NetTangibleAssets": "Net Tangible Assets",
    "WorkingCapital": "Working Capital",
    "InvestedCapital": "Invested Capital",
    "TangibleBookValue": "Tangible Book Value",
    "TotalDebt": "Total Debt",
    "NetDebt": "Net Debt",
    "ShareIssued": "Share Issued",
    "OrdinarySharesNumber": "Ordinary Shares Number",
    "TreasurySharesNumber": "Treasury Shares Number",
    "TotalCapitalization": "Total Capitalization",
    "CapitalLeaseObligations": "Capital Lease Obligations",

    # --- [Operating Activities (영업활동 현금흐름)] ---
    "OperatingCashFlow": "Total Cash From Operating Activities",
    "CashFlowFromContinuingOperatingActivities": "Cash Flow From Continuing Operating Activities",
    "NetIncomeFromContinuingOperations": "Net Income From Continuing Operations",
    "DepreciationAndAmortization": "Depreciation and Amortization",
    "DepreciationAmortizationDepletion": "Depreciation, Amortization & Depletion",
    "AssetImpairmentCharge": "Asset Impairment Charge",
    "StockBasedCompensation": "Stock Based Compensation",
    "OtherNonCashItems": "Other Non Cash Items",
    "OperatingGainsLosses": "Operating Gains and Losses",
    "ChangeInWorkingCapital": "Change In Working Capital",
    "ChangeInReceivables": "Change In Receivables",
    "ChangeInInventory": "Change In Inventory",
    "ChangeInPayablesAndAccruedExpense": "Change In Payables and Accrued Expense",
    "ChangeInOtherCurrentLiabilities": "Change In Other Current Liabilities",
    "ChangeInOtherWorkingCapital": "Change In Other Working Capital",

    # --- [Investing Activities (투자활동 현금흐름)] ---
    "InvestingCashFlow": "Total Cash From Investing Activities",
    "CashFlowFromContinuingInvestingActivities": "Cash Flow From Continuing Investing Activities",
    "CapitalExpenditure": "Capital Expenditure",
    "CapitalExpenditureReported": "Capital Expenditure Reported",
    "NetInvestmentPurchaseAndSale": "Net Investment Purchase and Sale",
    "PurchaseOfInvestment": "Purchase of Investment",
    "SaleOfInvestment": "Sale of Investment",
    "NetBusinessPurchaseAndSale": "Net Business Purchase and Sale",
    "SaleOfBusiness": "Sale of Business",
    "NetOtherInvestingChanges": "Net Other Investing Changes",

    # --- [Financing Activities (재무활동 현금흐름)] ---
    "FinancingCashFlow": "Total Cash From Financing Activities",
    "CashFlowFromContinuingFinancingActivities": "Cash Flow From Continuing Financing Activities",
    "NetIssuancePaymentsOfDebt": "Net Issuance Payments of Debt",
    "IssuanceOfDebt": "Issuance of Debt",
    "LongTermDebtIssuance": "Long Term Debt Issuance",
    "NetLongTermDebtIssuance": "Net Long Term Debt Issuance",
    "RepaymentOfDebt": "Repayment of Debt",
    "LongTermDebtPayments": "Long Term Debt Payments",
    "NetCommonStockIssuance": "Net Common Stock Issuance",
    "ProceedsFromStockOptionExercised": "Proceeds From Stock Option Exercised",
    "CommonStockPayments": "Common Stock Payments",
    "RepurchaseOfCapitalStock": "Repurchase of Capital Stock",
    "CashDividendsPaid": "Cash Dividends Paid",
    "CommonStockDividendPaid": "Common Stock Dividend Paid",
    "NetOtherFinancingCharges": "Net Other Financing Charges",

    # --- [Cash Summary & Supplemental (현금 변동 및 기타)] ---
    "EffectOfExchangeRateChanges": "Effect of Exchange Rate Changes",
    "ChangesInCash": "Changes In Cash",
    "BeginningCashPosition": "Beginning Cash Position",
    "EndCashPosition": "End Cash Position",
    "FreeCashFlow": "Free Cash Flow",
    "InterestPaidSupplementalData": "Interest Paid (Supplemental)",
    "IncomeTaxPaidSupplementalData": "Income Tax Paid (Supplemental)"
}

# 1. Finviz 기본 제공 지표 중 스코어링에 쓸 지표
BASIC_METRICS = [
    "P/E", "Forward P/E", "PEG", "P/S", "P/B", "P/Cash", "P/Free Cash Flow",
    "Dividend Yield", "EPS Growth Next Year", "ROE", "ROA", "ROI", 
    "Gross Margin", "Operating Margin", "Profit Margin", "Debt/Equity", 
    "Current Ratio", "Relative Strength Index (14)", "Volatility (Month)", "Short Float"
]

# 2. calculate_advanced_metrics 함수에서 우리가 직접 만든 지표
ADVANCED_METRICS = [
    "Momentum_Custom", "Momentum_12M_1M", "Momentum_6M_1M", "RSI_Volatility_Adj",
    "Short_Term_Accel", "MA_Convergence", "FCF_Yield", "GP_A_Quality",
    "GPA_Score_Internal", "PEG_Debt_Adj", "Net_Working_Capital_Value",
    "Inst_Inside_Buy", "Short_Squeeze_Potential", "Risk_Adj_Return", "Sales_Growth_Proxy"
]

# 3. 사이드바에서 보여줄 최종 통합 리스트 (알파벳 순 정렬)
ALL_STRATEGY_METRICS = sorted(BASIC_METRICS + ADVANCED_METRICS)

@st.cache_data
def get_sp500_domain_mapping():
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        table = pd.read_html(url)
        df = table[0]
        mapping = {}
        for _, row in df.iterrows():
            ticker = row['Symbol']
            company = row['Security']
            clean_name = company.split()[0].replace(',', '').replace('.', '').lower()
            mapping[ticker] = f"{clean_name}.com"
        
        # 대표적인 예외들만 살짝 보정
        mapping.update({"GOOGL": "google.com", "GOOG": "google.com", "NVDA": "nvidia.com", "META": "meta.com"})
        return mapping
    except:
        return {}

# 미리 매핑 데이터 생성 (한 번만 실행됨)
sp500_mapping = get_sp500_domain_mapping()


# --- 2. 유틸리티 함수 ---
def safe_num(series):
    return pd.to_numeric(series.astype(str).str.replace('%', '').str.replace(',', '').str.replace('$', '').replace('-', '0'), errors='coerce').fillna(0)

def get_column_config(df):
    config = {
        # 기본 정보
        "Rank": st.column_config.NumberColumn("Rank", width="small"),
        "Ticker": st.column_config.TextColumn("Ticker", width="small"),
        "Company": st.column_config.TextColumn("Company", width="medium"),
        
        # 스코어 및 커스텀 지표
        "Total_Score": st.column_config.ProgressColumn("Total_Score", min_value=0, max_value=100, format="%.2f"),
        "GPA_Score_Internal": st.column_config.NumberColumn("GPA_Quality", format="%.2f"),
        "Momentum_Custom": st.column_config.NumberColumn("Mom_Custom", format="%.2f"),
        
        # 가격 및 시총 (전처리에서 축약했으므로 TextColumn)
        "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
        "Market Cap": st.column_config.TextColumn("Market Cap", help="축약형 시총"),
        "Volume": st.column_config.NumberColumn("Volume", format="%d"),
        
        # 수익률 및 성장성 (모든 항목에 100을 곱했으므로 %.2f%% 적용)
        "Change": st.column_config.NumberColumn("Change", format="%.2f%%"),
        "Performance (Week)": st.column_config.NumberColumn("Perf (W)", format="%.2f%%"),
        "Performance (Month)": st.column_config.NumberColumn("Perf (M)", format="%.2f%%"),
        "Performance (Quarter)": st.column_config.NumberColumn("Perf (Q))", format="%.2f%%"),
        "Performance (Half Year)": st.column_config.NumberColumn("Perf (H)", format="%.2f%%"),
        "Performance (Year)": st.column_config.NumberColumn("Perf (Y)", format="%.2f%%"),
        "Performance (YTD)": st.column_config.NumberColumn("Perf YTD", format="%.2f%%"),
        
        # 수익성 지표
        "FCF_Yield": st.column_config.NumberColumn("FCF Yield", format="%.2f%%"),
        "Return on Equity": st.column_config.NumberColumn("ROE", format="%.2f%%"),
        "Return on Assets": st.column_config.NumberColumn("ROA", format="%.2f%%"),
        "Gross Margin": st.column_config.NumberColumn("Gross M", format="%.2f%%"),
        
        # 밸류에이션 (소수점 둘째자리)
        "P/E": st.column_config.NumberColumn("P/E", format="%.2f"),
        "Forward P/E": st.column_config.NumberColumn("Fwd P/E", format="%.2f"),
        "PEG": st.column_config.NumberColumn("PEG", format="%.2f"),
        "P/S": st.column_config.NumberColumn("P/S", format="%.2f"),
        
        # 수급 및 리스크
        "Short Float": st.column_config.TextColumn("Short Float"), # 원본 "1.12%" 형태 유지
        "Relative Strength Index (14)": st.column_config.NumberColumn("RSI", format="%d"),
    }
    
    # 리스트에 없는 나머지 모든 컬럼에 대해 기본 포맷 적용 (선택 사항)
    for col in df.columns:
        if col not in config:
            if df[col].dtype == 'float64':
                config[col] = st.column_config.NumberColumn(col, format="%.2f")
                
    return config

def format_market_cap(val):
    """지수 형태나 큰 숫자를 T, B, M으로 축약"""
    try:
        # 이미 문자열로 들어올 경우를 대비해 전처리
        if isinstance(val, str):
            val = val.replace('$', '').replace(',', '')
        val = float(val)
        if val >= 1e9: return f"{val/1e9:.2f}B"
        if val >= 1e6: return f"{val/1e6:.2f}M"
        return f"{val:,.0f}"
    except:
        return val

# --- 3. 데이터 수집 로직 (유연한 필터링 버전) ---
import concurrent.futures
import concurrent.futures
import streamlit as st
import time
import random

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(selected_cap, selected_sectors, selected_industries, excluded_countries):
    view_names = ["Overview", "Valuation", "Financial", "Ownership", "Performance", "Technical"]
    views = [Overview(), Valuation(), Financial(), Ownership(), Performance(), Technical()]
    
    filters_dict = {'Market Cap.': selected_cap}
    if selected_sectors and len(selected_sectors) == 1:
        filters_dict['Sector'] = selected_sectors[0]

    results = []
    
    progress_text = "🛡️ 서버 안전 모드로 데이터 수집 중..."
    my_bar = st.progress(0, text=progress_text)
    status_text = st.empty()

    def fetch_view(v_obj, name):
        # [핵심] 429 방지를 위해 요청 전후로 아주 짧은 랜덤 지연 추가
        time.sleep(random.uniform(0.5, 1.5)) 
        v_obj.set_filter(filters_dict=filters_dict)
        data = v_obj.screener_view()
        return data, name

    # [수정] max_workers를 2~3 정도로 낮춰서 Finviz의 경계망을 피합니다.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_to_view = {executor.submit(fetch_view, v, n): n for v, n in zip(views, view_names)}
        
        completed_count = 0
        for future in concurrent.futures.as_completed(future_to_view):
            try:
                completed_count += 1
                data, name = future.result()
                results.append(data)
                
                progress_val = completed_count / len(views)
                my_bar.progress(progress_val, text=f"{progress_text} ({completed_count}/{len(views)})")
                status_text.write(f"✅ 수집 완료: **{name}**")
            except Exception as e:
                st.error(f"❌ {future_to_view[future]} 수집 중 오류: {e}")
                results.append(None) # 에러 나도 루프는 유지

    time.sleep(1) # 마지막 데이터 처리 전 잠시 대기
    my_bar.empty()
    status_text.empty()

    # --- 이후 데이터 병합 로직은 동일 ---
    combined_df = None
    for df in results:
        if df is None or df.empty: continue
        if combined_df is None:
            combined_df = df
        else:
            combined_df = pd.merge(combined_df, df, on='Ticker', how='outer', suffixes=('', '_remove'))
            combined_df.drop(columns=[c for c in combined_df.columns if c.endswith('_remove')], inplace=True)
    
    if combined_df is not None:
        combined_df.rename(columns=FINVIZ_COL_MAP, inplace=True)
        if selected_sectors:
            combined_df = combined_df[combined_df['Sector'].isin(selected_sectors)]
        if selected_industries:
            combined_df = combined_df[combined_df['Industry'].isin(selected_industries)]
        if excluded_countries:
            combined_df = combined_df[~combined_df['Country'].isin(excluded_countries)]
        combined_df = combined_df.reset_index(drop=True)
            
    return combined_df

def calculate_advanced_metrics(df):
    temp_df = df.copy()
    
    # [방어 로직] 데이터 부재 시 0 처리
    def get_col(name):
        return safe_num(temp_df[name]) if name in temp_df.columns else pd.Series(0, index=temp_df.index)

    # --- 1. 데이터 추출 ---
    perf_year = get_col('Performance (Year)')
    perf_half = get_col('Performance (Half Year)')
    perf_month = get_col('Performance (Month)')
    perf_week = get_col('Performance (Week)')
    rsi = get_col('Relative Strength Index (14)')
    vol_m = get_col('Volatility (Month)')
    gross_m = get_col('Gross Margin')
    roa = get_col('Return on Assets')
    eps_next_y = get_col('EPS Growth Next Year')
    eps_this_y = get_col('EPS Growth This Year')
    eps_next_5y = get_col('EPS Growth Next Year')
    fcf_p = get_col('P/Free Cash Flow')
    peg = get_col('PEG')
    debt_eq = get_col('Total Debt/Equity')
    curr_r = get_col('Current Ratio')
    pb = get_col('P/B')
    inst_t = get_col('Institutional Transactions')
    insid_t = get_col('Insider Transactions')
    short_f = get_col('Short Float')
    rel_vol = get_col('Relative Volume')
    sales_5y = get_col('Sales Growth Past 5 Years')
    sma20 = get_col('20-Day Simple Moving Average')
    sma50 = get_col('50-Day Simple Moving Average')
    oper_m = get_col('Oper Margin')
    gross_m = get_col('Gross Margin')
    profit_m = get_col('Profit Margin')

    # --- 2. 모멘텀 계열 계산 (복구 완료) ---
    # 1년 수익률 부재 시 6개월로 대체
    main_perf = perf_year.replace(0, np.nan).combine_first(perf_half.replace(0, np.nan)).fillna(0)
    
    # 기하적 나눗셈 방식 (최근 1개월 상승분을 제외한 순수 추세)
    temp_df['Momentum_Custom'] = (((1 + main_perf/100) / (1 + perf_month/100) - 1) * 100).fillna(0)
    temp_df['Momentum_12M_1M'] = perf_year - perf_month
    temp_df['Momentum_6M_1M'] = perf_half - perf_month
    temp_df['Short_Term_Accel'] = perf_month - perf_week
    
    # RSI 및 변동성 조절 지표
    temp_df['RSI_Volatility_Adj'] = (rsi / (vol_m + 0.1)).fillna(0)
    
    # MA Convergence (inf 방어 로직 포함)
    convergence = (sma20 / (sma50 + 0.001))
    temp_df['MA_Convergence'] = convergence.replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.5, 2.0)

    # --- 3. 가치/퀄리티 계열 (복구 완료) ---
    # FCF Yield (극단적 값 방지를 위해 clip 처리)
    temp_df['FCF_Yield'] = np.where(fcf_p > 0, 1 / (fcf_p + 0.1), -0.5).clip(-0.5, 0.5)
    
    # GPA 및 미래 예상 GPA
    temp_df['GP_A_Quality'] = gross_m * roa
   
    # 부채 조절형 PEG 및 가치 지표
    # PEG가 0 이하(적자)면 50으로 치환하여 하위권으로 배치
    safe_peg = np.where(peg <= 0, 50, peg)
    temp_df['PEG_Debt_Adj'] = (safe_peg * (debt_eq + 1)).clip(0, 100)
    temp_df['Net_Working_Capital_Value'] = (curr_r / (pb + 0.1)).clip(0, 50)

    # --- 4. 리스크/수급/성장 (복구 완료) ---
    temp_df['Inst_Inside_Buy'] = inst_t + insid_t
    temp_df['Short_Squeeze_Potential'] = (short_f * rel_vol).clip(0, 500)
    temp_df['Risk_Adj_Return'] = (perf_year / (vol_m + 0.1)).fillna(0)
    
    # 성장성 대용 지표
    p_recent = (perf_month + perf_week) / 2
    #temp_df['Sales_Growth_Proxy'] = (sales_5y * 0.3) + (eps_next_y * 0.4) + (eps_next_5y * 0.3)
    Sales_Growth_Proxy = (sales_5y * 0.5) + (eps_next_y * 0.3) + (eps_this_y * 0.2)
    temp_df['Sales_Growth_Proxy'] = (sales_5y * 0.5) + (eps_next_y * 0.3) + (eps_this_y * 0.2)
    Margin_Quality = (0.5 + 0.3 * (oper_m / (gross_m + 0.01)) + 0.2 * (profit_m / (oper_m + 0.01)))
    temp_df['Margin_Quality'] = (0.5 + 0.3 * (oper_m / (gross_m + 0.01)) + 0.2 * (profit_m / (oper_m + 0.01)))
    temp_df['Estimated_Fwd_GP'] = (gross_m * (1 + Sales_Growth_Proxy / 100) * Margin_Quality)

    # GPA 스위칭 로직 (내부 랭킹 기반)
    ga_rank = temp_df['GP_A_Quality'].rank(pct=True)
    fwd_rank = temp_df['Estimated_Fwd_GP'].rank(pct=True)
    gpa_cond = ((temp_df['GP_A_Quality'] <= 0) & (eps_next_y > 15) & (temp_df['FCF_Yield'] > -0.15))
    temp_df['GPA_Score_Internal'] = np.where(gpa_cond, fwd_rank, ga_rank)

    return temp_df

def apply_v2_scoring(df, use_custom=False, custom_weights=None):
    if df is None or df.empty: 
        return df
    
    temp_df = df.copy()

    # [가이드] 낮을수록 좋은 지표 (역순 랭킹 적용 대상)
    low_better_metrics = [
        "P/E", "Forward P/E", "PEG", "P/S", "P/B", "P/Cash", "P/Free Cash Flow", 
        "LT Debt/Equity", "Total Debt/Equity", "Short Float", "Short Ratio", 
        "Volatility (Month)", "Volatility (Week)", "Average True Range", "Beta", 
        "Relative Strength Index (14)", "PEG_Debt_Adj", "Debt/Equity"
    ]

    # 스코어 초기화
    temp_df['Total_Score'] = 0.0

    # --- [CASE 1] 커스텀 전략 사용 시 ---
    if use_custom and custom_weights:
        for metric, weight in custom_weights.items():
            if metric in temp_df.columns:
                # 데이터를 수치화 (문자열 등이 섞여있을 경우 대비)
                valid_series = pd.to_numeric(temp_df[metric], errors='coerce').fillna(0)
                is_low_better = metric in low_better_metrics
                
                # 가중치를 적용하여 스코어 합산 (weight는 이미 사이드바에서 비율화됨)
                temp_df['Total_Score'] += valid_series.rank(ascending=not is_low_better, pct=True) * (weight * 100)

    # --- [CASE 2] 기존 섹터별 자동 전략 사용 시 ---
    else:
        # 1. STRATEGIES에 정의된 섹터별 가중치 적용
        for sector, weights in STRATEGIES.items():
            mask = temp_df['Sector'] == sector
            if not mask.any(): 
                continue
            
            for metric, weight in weights.items():
                if metric in temp_df.columns:
                    valid_series = pd.to_numeric(temp_df.loc[mask, metric], errors='coerce').fillna(0)
                    is_low_better = metric in low_better_metrics
                    temp_df.loc[mask, 'Total_Score'] += valid_series.rank(ascending=not is_low_better, pct=True) * weight
        
        # 2. 전략이 정의되지 않은 기타 섹터 처리 (GPA + Momentum)
        other_mask = ~temp_df['Sector'].isin(STRATEGIES.keys())
        if other_mask.any():
            # 안전하게 데이터 확보
            mom_rank = temp_df.loc[other_mask, 'Momentum_Custom'].rank(pct=True)
            gpa_rank = temp_df.loc[other_mask, 'GPA_Score_Internal'].rank(pct=True)
            
            temp_df.loc[other_mask, 'Total_Score'] = (mom_rank * 50) + (gpa_rank * 50)

    # 점수 순으로 정렬하여 반환
    return temp_df.sort_values("Total_Score", ascending=False)


# --- 7. 상단 필터 및 설정 UI (사이드바 대체) ---
# 페이지 최상단에 배치하여 사이드바 없이 넓게 사용
st.subheader("🔍 퀀트 스크리너 설정")

# 설정을 접었다 폈다 할 수 있는 Expander 사용
with st.expander("⚙️ 분석 조건 및 커스텀 전략 설정", expanded=True):
    # 1. 기본 필터 레이아웃 (3컬럼)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### 🛡️ 기본 필터")
        cap_option = st.selectbox("Market Cap", 
                                 ["Mega ($200bln and more)", "Large ($10bln to $200bln)", "Mid ($2bln to $10bln)", "Small ($300mln to $2bln)", "Micro ($50mln to $300mln)", "Nano (under $50mln)", "+Mega (over $200bln)", "+Large (over $10bln)", "+Mid (over $2bln)", "+Small (over $300mln)", "+Micro (over $50mln)", "-Large (under $200bln)", "-Mid (under $10bln)", "-Small (under $2bln)", "-Micro (under $300mln)"])
        excluded_countries = st.multiselect("제외 국가", ["China", "Argentina", "Australia", "Belgium", "Brazil", "Canada", "Chile", "China", "Colombia", "Denmark", "Finland", "France", "Germany", "Greece", "Hong Kong", "Hungary", "Iceland", "India", "Indonesia", "Ireland", "Israel", "Italy", "Japan", "Kazakhstan", "Luxembourg", "Malaysia", "Mexico", "Monaco", "Netherlands", "New Zealand", "Norway", "Peru", "Philippines", "Portugal", "Singapore", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Taiwan", "Turkey", "United Arab Emirates", "United Kingdom", "Uruguay", "USA", "Vietnam"])

    with col2:
        st.markdown("##### 🏢 섹터/산업")
        selected_sectors = st.multiselect("섹터 선택", list(SECTOR_INDUSTRY_MAP.keys()))
        
        available_industries = []
        if selected_sectors:
            for s in selected_sectors:
                available_industries.extend(SECTOR_INDUSTRY_MAP.get(s, []))
            available_industries = sorted(list(set(available_industries)))
        
        selected_industries = st.multiselect("산업 선택", available_industries, disabled=not selected_sectors)

    with col3:
        st.markdown("##### 🎯 전략 모드")
        use_custom_strategy = st.checkbox("나만의 전략 활성화", value=False, 
                                        help="섹터별 자동 전략 대신 직접 지표와 가중치를 설정합니다.")
        if not use_custom_strategy:
            st.info("💡현재 **섹터별 자동 최적화 전략** 적용 중입니다.")

    # 2. 커스텀 전략 상세 설정 (체크박스 활성화 시에만 표시)
    custom_weights = {}
    if use_custom_strategy:
        st.divider()
        st.subheader("⚖️ 커스텀 가중치 설계")
        
        selected_metrics = st.multiselect(
            "분석 지표 선택", 
            options=ALL_STRATEGY_METRICS,
            help="기본 재무 지표와 직접 계산된 Advanced 지표를 조합할 수 있습니다."
        )
        
        if selected_metrics:
            # 지표 입력창을 4개씩 가로로 배치하여 공간 절약
            metric_cols = st.columns(4)
            total_input_w = 0
            for i, m in enumerate(selected_metrics):
                with metric_cols[i % 4]:
                    w = st.number_input(f"{m} (%)", min_value=0, max_value=100, value=0, step=5, key=f"w_{m}")
                    custom_weights[m] = w
                    total_input_w += w
            
            # 가중치 검증 로직 (기존과 동일)
            if total_input_w == 0:
                st.warning("⚠️ 지표만 선택 시 '동일 가중치'가 적용됩니다.")
                for m in selected_metrics: custom_weights[m] = 1.0 / len(selected_metrics)
            elif total_input_w != 100:
                st.info(f"합계 {total_input_w}% → 자동으로 100% 비중 조절됨")
                for m in selected_metrics: custom_weights[m] = custom_weights[m] / total_input_w
            else:
                st.success("✅ 가중치 합계 100% 완료")
                for m in selected_metrics: custom_weights[m] = custom_weights[m] / 100.0

    # 3. 실행 버튼 (설정창 내부 하단 배치)
    st.divider()
    run_btn = st.button("🚀 스코어링 분석 실행", use_container_width=True, type="primary")


# --- 8. 결과 출력 로직 (중복 컬럼 및 지표 누락 완전 해결) ---
if run_btn:
    raw_df = fetch_data(cap_option, selected_sectors, selected_industries, excluded_countries)
    if raw_df is not None and not raw_df.empty:
        # 1. 계산용 숫자 데이터 생성 (지표가 추가된 데이터프레임)
        # 중요: calculate_advanced_metrics 안에서 이미 raw_df의 데이터가 수치화되어 포함됨
        processed_df = calculate_advanced_metrics(raw_df)
        
        # 2. 스코어링 적용
        scored_df = apply_v2_scoring(processed_df, use_custom_strategy, custom_weights)
        
        # 3. [해결책] 중복 방지 병합 로직
        # raw_df에는 문자열(예: "10.5%")이 있고, scored_df에는 계산된 숫자와 커스텀 지표가 있음.
        # 따라서, scored_df에서 커스텀 지표와 스코어만 추출해서 raw_df에 붙입니다.
        
        # 우리가 만든 커스텀 지표 리스트 (calculate_advanced_metrics에서 생성한 것들)
        custom_metrics = [
            'Momentum_Custom', 'Momentum_12M_1M', 'Momentum_6M_1M', 
            'RSI_Volatility_Adj', 'Short_Term_Accel', 'MA_Convergence', 
            'FCF_Yield', 'GP_A_Quality', 'Estimated_Fwd_GP', 'Margin_Quality',
            'GPA_Score_Internal', 'PEG_Debt_Adj', 'Net_Working_Capital_Value', 
            'Inst_Inside_Buy', 'Short_Squeeze_Potential', 'Risk_Adj_Return', 'Sales_Growth_Proxy'
        ]
        
        # 실제로 데이터프레임에 존재하는 커스텀 지표만 필터링
        existing_custom = [m for m in custom_metrics if m in scored_df.columns]
        
        # 병합용 데이터 준비: Ticker + 스코어 + 커스텀 지표
        merge_subset = scored_df[['Ticker', 'Total_Score'] + existing_custom]
        
        # 중복 방지를 위해 raw_df에서 이미 존재하는 커스텀 지표명이 있다면 삭제 후 병합
        clean_raw_df = raw_df.drop(columns=[c for c in existing_custom if c in raw_df.columns], errors='ignore')
        
        final_display = pd.merge(clean_raw_df, merge_subset, on='Ticker', how='left')
        final_display.rename(columns=FINVIZ_COL_MAP, inplace=True)
        final_display['Market Cap'] = final_display['Market Cap'].apply(format_market_cap)

        # B. 소수(0.0787)를 퍼센트(7.87)로 변환할 대상 리스트
        # [최종] 100을 곱해서 % 단위로 만들 항목 리스트
        pct_to_100_cols = [
            # 1. 수익률 (Performance) 계열 - 신규 추가 및 명칭 수정
            'Performance (Week)', 'Performance (Month)', 'Performance (Quarter)', 'Performance (Half Year)', 'Performance (Year)', 'Performance (YTD)', 'Change', 
            # 2. 수익성 및 퀄리티 (Quality) 계열
            'Return on Assets', 'Return on Equity', 'Gross Margin', 'Oper Margin', 'Profit Margin','FCF_Yield', 'Dividend Yield', 
            # 3. 기술적 지표 및 변동성 (Technicals)
            'Volatility (Week)', 'Volatility (Month)','20-Day Simple Moving Average', '50-Day Simple Moving Average', '200-Day Simple Moving Average', '52W High', '52W Low',
            # 4. 수급 및 기타
            'Inst_Inside_Buy'
        ]
        
        for col in pct_to_100_cols:
            if col in final_display.columns:
                # safe_num을 거쳐서 숫자로 만든 뒤 100을 곱함
                final_display[col] = safe_num(final_display[col]) * 100

        # C. 나머지 수치형 데이터 강제 형변환 (NumberColumn 오류 방지)
        numeric_cols = ['Total_Score', 'Price', 'Volume', 'Avg Volume', 'P/E', 'Forward P/E', 'PEG', 'P/S', 'P/B', 'P/C', 'P/Free Cash Flow']
        for col in numeric_cols:
            if col in final_display.columns:
                final_display[col] = safe_num(final_display[col])

        st.session_state.final_df = final_display.sort_values("Total_Score", ascending=False)

# --- 9. 결과 출력부 ---
if 'final_df' in st.session_state and st.session_state.final_df is not None:
    final_df = st.session_state.final_df
    
    st.success("✅ 분석 완료! 표의 행(Row)을 클릭하면 하단에 상세 정보가 나타납니다.")
    
    # ---------------- 여기서부터 교체 (섹터별 전략 동적 표시) ----------------
    with st.expander("📖 전략 정보(사용 지표 및 가중치)", expanded=False):
        if use_custom_strategy and custom_weights:
            # --- [1] 커스텀 전략 (나만의 전략) ---
            active_weights = {k: v * 100 for k, v in custom_weights.items() if v > 0}
            if active_weights:
                st.markdown("#### 🛠️ 커스텀 가중치 분석")
                df_weights = pd.DataFrame([active_weights]).T.reset_index()
                df_weights.columns = ['지표', '비중(%)']
                
                fig = px.pie(df_weights, values='비중(%)', names='지표', 
                             hole=0.45, title="현재 적용된 커스텀 전략",
                             color_discrete_sequence=px.colors.qualitative.Pastel)
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(height=400, margin=dict(t=50, b=20, l=20, r=20))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ 설정된 커스텀 가중치가 없습니다.")

        else:
            # --- [2] 일반 섹터 모드 (무조건 탭 활용) ---
            target_sectors = selected_sectors if selected_sectors else list(STRATEGIES.keys())
            
            # 섹터 이름을 기반으로 탭 생성 (모바일에서 가로 스크롤 가능)
            st.markdown(f"#### 🌐 섹터별 자동 가중치 분석")
            sector_tabs = st.tabs(target_sectors)
            
            for i, s_name in enumerate(target_sectors):
                with sector_tabs[i]:
                    s_logic = STRATEGIES.get(s_name, {})
                    df_s = pd.DataFrame([s_logic]).T.reset_index()
                    df_s.columns = ['지표', '비중(%)']
                    
                    # 2컬럼 레이아웃: 왼쪽(차트), 오른쪽(상세 수치 리스트)
                    col1, col2 = st.columns([1.2, 1])
                    
                    with col1:
                        fig = px.pie(df_s, values='비중(%)', names='지표', 
                                     hole=0.5, 
                                     color_discrete_sequence=px.colors.qualitative.Safe)
                        fig.update_layout(
                            showlegend=False, 
                            height=300, 
                            margin=dict(t=10, b=10, l=10, r=10),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)'
                        )
                        fig.update_traces(textinfo='percent+label', textfont_size=11)
                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                    
                    with col2:
                        st.markdown(f"**📍 {s_name} 핵심 로직**")
                        # 가중치가 높은 순서대로 내림차순 정렬하여 텍스트 표시
                        sorted_items = sorted(s_logic.items(), key=lambda x: x[1], reverse=True)
                        for metric, weight in sorted_items:
                            # 가중치가 높은 지표는 강조(bold)
                            st.write(f"- {metric}: **{weight}%**")
                        
                        st.caption("해당 섹터의 특성을 반영한 가중치입니다.")
                        st.caption("나만의 전략 활성화 체크 시 전략을 만들 수 있습니다.")
    # ---------------- 여기까지 교체 ----------------

    main_tabs = st.tabs(["📊 통합 순위", "📁 섹터별 상세 순위"])

    # 공통 고정 컬럼 정의
    base_cols = ['Rank', 'Ticker', 'Company', 'Sector', 'Industry', 'Country', 'Market Cap', 'Total_Score']

    # --- Tab 1: 통합 순위 ---
    with main_tabs[0]:
        # Rank 생성 로직 삭제 및 Ticker 인덱스 설정
        display_df = final_df.head(50).copy().set_index('Ticker')

        current_config = get_column_config(display_df)        

        if use_custom_strategy and custom_weights:
            priority_metrics = [m for m in custom_weights.keys() if m in display_df.columns]
        else:
            priority_metrics = []
        
        # base_cols에서 Rank와 Ticker(인덱스) 제외하고 순서 배치
        display_cols = [c for c in base_cols if c not in ['Rank', 'Ticker']] + priority_metrics
        other_cols = [c for c in display_df.columns if c not in display_cols]
        display_df = display_df[display_cols + other_cols]

        def highlight_custom(s):
            return ['background-color: #f0f2f6' if s.name in priority_metrics else '' for _ in s]

        event = st.dataframe(
            display_df.style.apply(highlight_custom), 
            use_container_width=True,
            height=600,
            on_select="rerun",
            selection_mode="single-row", 
            column_config=current_config,
            key="main_table_final"
        )

        if event and len(event.get("selection", {}).get("rows", [])) > 0:
            selected_idx = event["selection"]["rows"][0]
            st.session_state.selected_ticker = display_df.index[selected_idx]

    # --- Tab 2: 섹터별 상세 순위 ---
    with main_tabs[1]:
        sector_tabs_list = selected_sectors if selected_sectors else list(STRATEGIES.keys())
        if sector_tabs_list:
            sub_tabs = st.tabs(sector_tabs_list)
            for i, sector_name in enumerate(sector_tabs_list):
                with sub_tabs[i]:
                    # Rank 생성 없이 바로 Ticker 인덱스 설정
                    sector_df = final_df[final_df['Sector'] == sector_name].head(30).copy().set_index('Ticker')
                    
                    if not sector_df.empty:
                        current_sector_config = get_column_config(sector_df)

                        if use_custom_strategy and custom_weights:
                            priority_metrics = [m for m in custom_weights.keys() if m in sector_df.columns]
                        else:
                            priority_metrics = [m for m in STRATEGIES.get(sector_name, {}).keys() if m in sector_df.columns]
                        
                        # 컬럼 재배치 (Rank 제외)
                        display_cols = [c for c in base_cols if c not in ['Rank', 'Ticker']] + priority_metrics
                        other_cols = [c for c in sector_df.columns if c not in display_cols]
                        sector_df = sector_df[display_cols + other_cols]

                        def highlight_priority(s):
                            color = '#e6ffed' if use_custom_strategy else '#f0faff'
                            return [f'background-color: {color}' if s.name in priority_metrics else '' for _ in s]

                        sec_event = st.dataframe(
                            sector_df.style.apply(highlight_priority), 
                            use_container_width=True,
                            height=600,
                            on_select="rerun",
                            selection_mode="single-row",
                            column_config=current_sector_config,
                            key=f"sec_table_{sector_name}"
                        )
                        
                        if sec_event and len(sec_event.get("selection", {}).get("rows", [])) > 0:
                            s_idx = sec_event["selection"]["rows"][0]
                            st.session_state.selected_ticker = sector_df.index[s_idx]
                    else:
                        st.info(f"{sector_name} 섹터에 해당하는 종목이 없습니다.")
                    
    st.divider()
    col_dl, _ = st.columns([1, 1])
    with col_dl:
        csv = final_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 전체 분석 결과 CSV 다운로드",
            data=csv,
            file_name=f"quant_result_{time.strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

import yfinance as yf

# --- [핵심] 수동 TTM 계산 및 DISPLAY_MAP 정렬이 포함된 리포트 ---
if st.session_state.get('selected_ticker'):
    target = st.session_state.selected_ticker
    st.divider()
    
    # [수정] 해당 티커가 final_df에 존재하는지 먼저 확인
    target_rows = final_df[final_df['Ticker'] == target]
    
    if target_rows.empty:
        st.warning(f"⚠️ {target}에 대한 요약 데이터를 찾을 수 없습니다. 필터 설정을 확인해주세요.")
    else:
        # 데이터가 있을 때만 iloc[0] 접근
        stock_info = target_rows.iloc[0]

        with st.spinner(f"📡 {target}의 데이터를 분석하고 DISPLAY_MAP에 맞춰 정렬 중..."):
            try:
                stock_obj = yf.Ticker(target)
                
                # 1. 상단 요약 정보
                with st.container(border=True):
                    stock_info = final_df[final_df['Ticker'] == target].iloc[0]
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        domain = None
                        
                        # [1순위] S&P 500 캐시 확인 (속도가 가장 빠름)
                        if target in sp500_mapping:
                            domain = sp500_mapping.get(target)

                        # [2순위] 야후 파이낸스 확인 (S&P 500에 없을 경우 실시간 조회)
                        if not domain:
                            try:
                                ticker_obj = yf.Ticker(target)
                                official_website = ticker_obj.info.get('website', '')
                                if official_website:
                                    domain = official_website.replace('http://', '').replace('https://', '').split('/')[0]
                            except:
                                pass

                        # [3순위] 최후의 추측 (회사 이름 기반)
                        if not domain:
                            clean_name = stock_info['Company'].split()[0].replace(',', '').lower()
                            domain = f"{clean_name}.com"

                        # [4순위] 구글 Favicon API 적용
                        logo_url = f"https://www.google.com/s2/favicons?domain={domain}&sz=128"

                        # [단계 3] 이미지 출력 및 [이미지 예외 처리]
                        st.markdown(
                            f"""
                            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 5px;">
                                <img src="{logo_url}" width="40" height="40" style="border-radius: 4px; object-fit: contain;" 
                                    onerror="this.src='https://cdn-icons-png.flaticon.com/512/2583/2583125.png'">
                                <h2 style="margin: 0; padding: 0; font-size: 28px;">{target} 상세 리포트</h2>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        st.write(f"**{stock_info['Company']}**")
                        st.metric("퀀트 스코어", f"{stock_info['Total_Score']:.2f}")
                    with col2:
                        st.write("🔗 **빠른 분석 링크**")
                        # 버튼들을 더 깔끔하게 정렬 (가로 배치)
                        btn_col1, btn_col2, btn_col3 = st.columns(3)
                        with btn_col1:
                            st.link_button(f"📊 Finviz", f"https://finviz.com/quote.ashx?t={target}", use_container_width=True)
                        with btn_col2:
                            st.link_button(f"🔍 Yahoo", f"https://finance.yahoo.com/quote/{target}", use_container_width=True)
                        with btn_col3:
                            st.link_button(f"📈 TradingView", f"https://www.tradingview.com/symbols/{target}", use_container_width=True)

                # 2. 재무제표 상세 탭
                st.write(f"### 📑 {target} 전체 재무제표")
                st_tabs = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow"])
                
                statements_config = {
                    "Income Statement": stock_obj.get_income_stmt,
                    "Balance Sheet": stock_obj.get_balance_sheet,
                    "Cash Flow": stock_obj.get_cash_flow
                }

                # [스마트 포맷팅 함수] 소수점 지표(EPS 등)는 유지하고 큰 금액은 콤마 처리
                def smart_format(x):
                    if pd.isnull(x) or not isinstance(x, (int, float, np.number)):
                        return x
                    # 절대값이 100 이상이면 큰 금액으로 간주하여 정수 + 콤마 처리
                    if abs(x) >= 100:
                        return f"{int(x):,}"
                    # 100 미만이면 EPS, 비율 등으로 간주하여 소수점 2자리 유지
                    else:
                        if x == int(x): return f"{int(x)}"
                        return f"{x:,.2f}"

                # [지표 처리 함수] Trend와 TTM을 가장 왼쪽에 배치
                def process_with_all_features(raw_df, ttm_data=None, is_balance_sheet=False):
                    if raw_df is None or raw_df.empty:
                        return None
                    
                    df = raw_df.copy()
                    df.index = df.index.str.replace(" ", "")
                    
                    # 1. DISPLAY_MAP 순서에 따른 필터링 및 정렬
                    existing_keys = [k for k in DISPLAY_MAP.keys() if k in df.index]
                    if not existing_keys: return None
                    df = df.reindex(existing_keys)
                    
                    # 2. [데이터 준비] TTM 유무에 따른 그래프 리스트 생성
                    has_ttm = False
                    if not is_balance_sheet and ttm_data is not None:
                        has_ttm = True
                        # 과거 실적(뒤집기) + TTM 값 포함하여 Trend 생성
                        past_values = df.iloc[:, ::-1]
                        trend_list = []
                        for idx, row in past_values.iterrows():
                            # 해당 지표의 TTM 값을 가져와 리스트 마지막에 추가
                            ttm_val = ttm_data.get(idx, 0)
                            combined = row.fillna(0).tolist() + [ttm_val]
                            trend_list.append(combined)
                        
                        # 3. 컬럼 삽입 (Trend를 0번에, TTM을 1번에 삽입)
                        df.insert(0, 'Trend', trend_list)
                        df.insert(1, 'Current TTM', ttm_data)
                    else:
                        # Balance Sheet 등 TTM이 없는 경우 Trend만 0번에 삽입
                        df.insert(0, 'Trend', df.iloc[:, ::-1].fillna(0).values.tolist())

                    # 4. 이름 변경 및 포맷팅
                    df.index = [DISPLAY_MAP[k] for k in df.index]
                    data_cols = [c for c in df.columns if c != 'Trend']
                    formatted_df = df[data_cols].map(smart_format)
                    formatted_df.insert(0, 'Trend', df['Trend']) # Trend 열은 포맷팅 없이 원본 유지
                    
                    return formatted_df

                # --- [출력 루프] ---
                for idx, (name, func) in enumerate(statements_config.items()):
                    with st_tabs[idx]:
                        sub_tabs = st.tabs(["📅 Annual", "⏱ Quarterly"])
                        is_bs = (name == "Balance Sheet")
                        
                        # TTM 계산용 분기 데이터 미리 확보
                        df_q_raw = func(freq='quarterly')
                        df_y_raw = func(freq='yearly')
                        
                        calculated_ttm = None
                        if not is_bs and df_q_raw is not None and len(df_q_raw.columns) >= 4:
                            temp_q = df_q_raw.copy()
                            temp_q.index = temp_q.index.str.replace(" ", "")
                            calculated_ttm = temp_q.iloc[:, :4].sum(axis=1)

                        # 1. 연간 탭
                        with sub_tabs[0]:
                            processed_y = process_with_all_features(df_y_raw, calculated_ttm, is_bs)
                            if processed_y is not None:
                                st.dataframe(processed_y, column_config={
                                    "Trend": st.column_config.BarChartColumn("Trend", width="small"),
                                    "Current TTM": st.column_config.Column("Current TTM", width="small")
                                }, use_container_width=True, height=800)
                        
                        # 2. 분기 탭
                        with sub_tabs[1]:
                            processed_q = process_with_all_features(df_q_raw, calculated_ttm, is_bs)
                            if processed_q is not None:
                                st.dataframe(processed_q, column_config={
                                    "Trend": st.column_config.BarChartColumn("Trend", width="small"),
                                    "Current TTM": st.column_config.Column("Current TTM", width="small")
                                }, use_container_width=True, height=800)
                                
            except Exception as e:
                st.error(f"오류 발생: {e}")

    

    