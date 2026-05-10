"""
Sector Mutual Fund Analysis Tool
--------------------------------
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import hmean
import warnings
warnings.filterwarnings('ignore')

# ---------- File paths ----------
# These exactly match the files with spaces in your GitHub repo
NAV_FILES = ['sector funds 1.xlsx', 'secotr funds 2.xlsx',
             'secotr funds 3.xlsx', 'sector funds 4.xlsx']
SECTOR_ALLOC_FILE = 'secotrs aloocations.xlsx'
STOCK_ALLOC_FILE  = 'sectors stock alocation.xlsx'
PE_FILE           = 'sector pe rstio.xlsx'

# ============================================================
# DATA LOADING FUNCTIONS (Cached for speed)
# ============================================================
@st.cache_data
def load_nav_data():
    all_navs = []
    for f in NAV_FILES:
        raw = pd.read_excel(f, header=None)
        fund_names = raw.iloc[2, 1:].tolist()            
        data = raw.iloc[4:, :].copy()                      
        data.columns = ['Date'] + fund_names
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        data = data.dropna(subset=['Date']).set_index('Date')
        data = data.loc[:, data.columns.notna()]
        data = data.apply(pd.to_numeric, errors='coerce')
        all_navs.append(data)
    nav = pd.concat(all_navs, axis=1, sort=True)
    nav = nav.loc[:, ~nav.columns.duplicated()]          
    nav = nav.sort_index()
    return nav

@st.cache_data
def rolling_returns_summary(nav):
    windows = {'1Y': 252, '3Y': 756, '5Y': 1260}
    out = {}
    for fund in nav.columns:
        s = nav[fund].dropna()
        row = {'Fund': fund, 'Data Points': len(s)}
        for label, w in windows.items():
            if len(s) > w:
                roll = (s / s.shift(w)) ** (252.0 / w) - 1
                roll = roll.dropna() * 100
                row[f'{label} Median Return (%)'] = round(roll.median(), 2) if len(roll) else np.nan
            else:
                row[f'{label} Median Return (%)'] = np.nan
        out[fund] = row
    return pd.DataFrame(out).T.reset_index(drop=True)

@st.cache_data
def load_pe_data():
    raw = pd.read_excel(PE_FILE, header=None)
    funds = {}
    current_fund = None
    rows = []
    for _, r in raw.iterrows():
        cell0 = str(r[0]) if pd.notna(r[0]) else ''
        if cell0.startswith('Scheme Name:'):
            if current_fund and rows:
                funds[current_fund] = pd.DataFrame(rows, columns=['Date', 'PE', 'PBV', 'DY', 'MCAP'])
            current_fund = cell0.replace('Scheme Name:', '').strip()
            rows = []
        else:
            try:
                d = pd.to_datetime(r[0])
                pe   = pd.to_numeric(r[1], errors='coerce')
                pbv  = pd.to_numeric(r[3], errors='coerce')
                dy   = pd.to_numeric(r[5], errors='coerce')
                mcap = pd.to_numeric(r[6], errors='coerce')
                if pd.notna(d):
                    rows.append([d, pe, pbv, dy, mcap])
            except Exception:
                pass
    if current_fund and rows:
        funds[current_fund] = pd.DataFrame(rows, columns=['Date', 'PE', 'PBV', 'DY', 'MCAP'])
    return funds

@st.cache_data
def pe_summary(pe_data):
    rows = []
    for fund, df in pe_data.items():
        pe = df['PE'].dropna()
        pe = pe[pe > 0]                                  
        if len(pe) == 0: continue
        rows.append({
            'Fund': fund,
            'Latest PE': round(df['PE'].dropna().iloc[0], 2) if df['PE'].dropna().size else np.nan,
            'PE (Harmonic Mean)': round(hmean(pe), 2),
            'PE (Arithmetic Mean)': round(pe.mean(), 2),
            'PE (Median)': round(pe.median(), 2),
            'PBV (HM)': round(hmean(df['PBV'].dropna()[df['PBV'] > 0]), 2) if (df['PBV'].dropna() > 0).any() else np.nan,
            'Months of Data': len(pe),
        })
    return pd.DataFrame(rows)

@st.cache_data
def load_sector_allocation():
    df = pd.read_excel(SECTOR_ALLOC_FILE, header=3)
    df.columns = ['Fund', 'Sector', 'No of Cos', 'Allocation (%)']
    df = df.dropna(subset=['Fund', 'Sector'])
    df['Allocation (%)'] = pd.to_numeric(df['Allocation (%)'], errors='coerce')
    return df

@st.cache_data
def load_stock_allocation():
    df = pd.read_excel(STOCK_ALLOC_FILE, header=3)
    df.columns = ['Fund', 'Company', 'Asset', 'Sector', 'Allocation (%)']
    df = df.dropna(subset=['Fund', 'Company'])
    df['Allocation (%)'] = pd.to_numeric(df['Allocation (%)'], errors='coerce')
    return df

def list_sectors(sector_df):
    return sorted(sector_df['Sector'].dropna().unique().tolist())

# ============================================================
# MAIN STREAMLIT APP
# ============================================================
def main():
    st.set_page_config(page_title="Sector Fund Analysis", layout="wide")
    st.title("📈 Sector Mutual Fund Analysis Tool")

    # This loading spinner will show while the Excel files are being read
    with st.spinner("Loading and processing data..."):
        nav = load_nav_data()
        returns = rolling_returns_summary(nav)
        pe_raw = load_pe_data()
        pe_summary_df = pe_summary(pe_raw)
        sector_df = load_sector_allocation()
        stock_df  = load_stock_allocation()

        # Clean fund names so merges work perfectly
        for d in (returns, pe_summary_df, sector_df, stock_df):
            d['Fund'] = d['Fund'].astype(str).str.strip()

    st.success("Data loaded successfully!")

    # ---------- UI Tabs ----------
    tab1, tab2, tab3 = st.tabs(["🏆 Top Funds", "📊 Sector Deep Dive", "🔍 Single Fund Overview"])

    with tab1:
        st.subheader("Top 10 Funds by 5Y Median Rolling Return")
        top5y = (returns.dropna(subset=['5Y Median Return (%)'])
                        .sort_values('5Y Median Return (%)', ascending=False)
                        .head(10))
        st.dataframe(top5y[['Fund', '1Y Median Return (%)', '3Y Median Return (%)', '5Y Median Return (%)']], use_container_width=True)

        st.subheader("All PE Summaries")
        st.dataframe(pe_summary_df, use_container_width=True)

    with tab2:
        sectors = list_sectors(sector_df)
        selected_sector = st.selectbox("Choose a Sector to Analyze", sectors)
        
        if selected_sector:
            st.subheader(f"Analysis for: {selected_sector}")
            sub = sector_df[sector_df['Sector'].str.lower() == selected_sector.lower()]
            sub = sub.sort_values('Allocation (%)', ascending=False).head(15).reset_index(drop=True)
            
            if sub.empty:
                st.warning("No funds with allocation in this sector.")
            else:
                sub = sub.merge(returns, on='Fund', how='left')
                sub = sub.merge(pe_summary_df[['Fund', 'PE (Harmonic Mean)', 'Latest PE']], on='Fund', how='left')
                cols = ['Fund', 'Allocation (%)', '1Y Median Return (%)', '3Y Median Return (%)', '5Y Median Return (%)', 'Latest PE', 'PE (Harmonic Mean)']
                cols = [c for c in cols if c in sub.columns]
                
                st.dataframe(sub[cols], use_container_width=True)
                
                avg_pe_hm = sub['PE (Harmonic Mean)'].dropna()
                if len(avg_pe_hm):
                    st.info(f"**Average PE (Harmonic Mean) across these funds:** {hmean(avg_pe_hm[avg_pe_hm > 0]):.2f}")

    with tab3:
        fund_list = returns['Fund'].unique().tolist()
        selected_fund = st.selectbox("Choose a Fund", fund_list)
        
        if selected_fund:
            st.subheader(f"Fund: {selected_fund}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Rolling Returns (Median, Annualised)**")
                r = returns[returns['Fund'] == selected_fund]
                if not r.empty:
                    st.dataframe(r[['1Y Median Return (%)', '3Y Median Return (%)', '5Y Median Return (%)']].T, use_container_width=True)
                
                st.write("**Valuation**")
                p = pe_summary_df[pe_summary_df['Fund'] == selected_fund]
                if not p.empty:
                    st.dataframe(p[['Latest PE', 'PE (Harmonic Mean)', 'PE (Median)', 'PBV (HM)']].T, use_container_width=True)

            with col2:
                st.write("**Top Sector Allocations**")
                sec = sector_df[sector_df['Fund'] == selected_fund]
                if not sec.empty:
                    st.dataframe(sec.sort_values('Allocation (%)', ascending=False).head(8)[['Sector', 'Allocation (%)']], use_container_width=True)
                
            st.write("**Top Stock Holdings**")
            stkt = stock_df[stock_df['Fund'] == selected_fund]
            if not stkt.empty:
                st.dataframe(stkt.sort_values('Allocation (%)', ascending=False).head(10)[['Company', 'Sector', 'Allocation (%)']], use_container_width=True)

if __name__ == '__main__':
    main()
