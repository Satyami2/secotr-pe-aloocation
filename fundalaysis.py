"""
Sector Mutual Fund Analysis Tool - Pro UI
----------------------------------------------
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import hmean
import warnings
warnings.filterwarnings('ignore')

# ---------- File paths ----------
NAV_FILES = ['sector funds 1.xlsx', 'secotr funds 2.xlsx',
             'secotr funds 3.xlsx', 'sector funds 4.xlsx']
SECTOR_ALLOC_FILE = 'secotrs aloocations.xlsx'
STOCK_ALLOC_FILE  = 'sectors stock alocation.xlsx'
PE_FILE           = 'sector pe rstio.xlsx'

# Assumption for Sharpe/Sortino calculations (6% annual)
RISK_FREE_RATE = 0.06 

# ============================================================
# DATA LOADING FUNCTIONS 
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
def compute_advanced_metrics(nav):
    windows = {'1Y': 252, '3Y': 756, '5Y': 1260}
    out = {}
    
    daily_ret = nav.pct_change()
    bench_ret = daily_ret.mean(axis=1) 
    up_days = bench_ret > 0
    down_days = bench_ret < 0
    bench_up_mean = bench_ret[up_days].mean()
    bench_down_mean = bench_ret[down_days].mean()

    for fund in nav.columns:
        s = nav[fund].dropna()
        fund_daily_ret = daily_ret[fund].dropna()
        row = {'Fund': fund}
        
        # 1. Rolling Returns
        for label, w in windows.items():
            if len(s) > w:
                roll = (s / s.shift(w)) ** (252.0 / w) - 1
                roll = roll.dropna() * 100
                row[f'{label} Roll Med (%)'] = round(roll.median(), 2) if len(roll) else np.nan
            else:
                row[f'{label} Roll Med (%)'] = np.nan

        row['1W Return (%)'] = round((s.iloc[-1] / s.iloc[-6] - 1) * 100, 2) if len(s) >= 6 else np.nan
        row['1M Return (%)'] = round((s.iloc[-1] / s.iloc[-22] - 1) * 100, 2) if len(s) >= 22 else np.nan

        # 2. Risk & Drawdowns
        roll_max = s.cummax()
        drawdowns = (s / roll_max - 1) * 100
        row['Max Drawdown (%)'] = round(drawdowns.min(), 2) if len(drawdowns) else np.nan

        fund_up_mean = fund_daily_ret[up_days].mean()
        fund_down_mean = fund_daily_ret[down_days].mean()
        row['Up Capture (%)'] = round((fund_up_mean / bench_up_mean) * 100, 2) if bench_up_mean else np.nan
        row['Down Capture (%)'] = round((fund_down_mean / bench_down_mean) * 100, 2) if bench_down_mean else np.nan

        # 3. Sharpe & Sortino (Annualized)
        if len(fund_daily_ret) > 252:
            ann_ret = fund_daily_ret.mean() * 252
            ann_vol = fund_daily_ret.std() * np.sqrt(252)
            row['Sharpe Ratio'] = round((ann_ret - RISK_FREE_RATE) / ann_vol, 2) if ann_vol != 0 else np.nan
            
            downside_ret = fund_daily_ret[fund_daily_ret < 0]
            down_vol = downside_ret.std() * np.sqrt(252)
            row['Sortino Ratio'] = round((ann_ret - RISK_FREE_RATE) / down_vol, 2) if down_vol != 0 else np.nan
        else:
            row['Sharpe Ratio'] = np.nan
            row['Sortino Ratio'] = np.nan

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
            'PBV (HM)': round(hmean(df['PBV'].dropna()[df['PBV'] > 0]), 2) if (df['PBV'].dropna() > 0).any() else np.nan,
        })
    return pd.DataFrame(rows)

@st.cache_data
def load_sector_allocation():
    df = pd.read_excel(SECTOR_ALLOC_FILE, header=3)
    df.columns = ['Fund', 'Sector', 'No of Cos', 'Allocation (%)']
    df = df.dropna(subset=['Fund', 'Sector'])
    df['Allocation (%)'] = pd.to_numeric(df['Allocation (%)'], errors='coerce')
    
    def assign_category(sec):
        sec = str(sec).lower()
        if any(x in sec for x in ['bank', 'financ', 'nbfc', 'insurance']): return 'Banks & Finance'
        if any(x in sec for x in ['it', 'tech', 'software', 'teck', 'computer']): return 'Tech'
        if any(x in sec for x in ['pharma', 'health', 'medical']): return 'Pharma & Healthcare'
        if any(x in sec for x in ['auto']): return 'Auto'
        if any(x in sec for x in ['energy', 'power', 'oil', 'gas', 'utilit']): return 'Energy & Power'
        if any(x in sec for x in ['fmcg', 'consum', 'retail']): return 'Consumption'
        if any(x in sec for x in ['infra', 'construct', 'capital goods', 'cement']): return 'Infrastructure'
        if any(x in sec for x in ['media', 'entertain']): return 'Media'
        if any(x in sec for x in ['service', 'telecom', 'hospital']): return 'Services'
        if any(x in sec for x in ['metal', 'mining', 'steel']): return 'Metals & Mining'
        return 'Other Sectors'
        
    df['Category'] = df['Sector'].apply(assign_category)
    return df

@st.cache_data
def load_stock_allocation():
    df = pd.read_excel(STOCK_ALLOC_FILE, header=3)
    df.columns = ['Fund', 'Company', 'Asset', 'Sector', 'Allocation (%)']
    df = df.dropna(subset=['Fund', 'Company'])
    df['Allocation (%)'] = pd.to_numeric(df['Allocation (%)'], errors='coerce')
    return df

# ============================================================
# MAIN STREAMLIT APP
# ============================================================
def main():
    st.set_page_config(page_title="Pro Fund Analysis", layout="wide")
    st.title("📈 Pro Mutual Fund Analysis & Screener")

    with st.spinner("Calculating metrics (Drawdowns, Sharpe, Sortino)..."):
        nav = load_nav_data()
        metrics_df = compute_advanced_metrics(nav)
        
        pe_raw = load_pe_data()
        pe_df = pe_summary(pe_raw)
        
        sector_df = load_sector_allocation()
        stock_df  = load_stock_allocation()

        for d in (metrics_df, pe_df, sector_df, stock_df):
            d['Fund'] = d['Fund'].astype(str).str.strip()

    master_df = metrics_df.merge(pe_df, on='Fund', how='left')

    # ---------- UI Tabs ----------
    tab1, tab2, tab3 = st.tabs(["🏆 All Funds & Rankings", "🗂️ Category Deep Dive", "⚖️ Fund Comparison"])

    # ---------------------------------------------------------
    # TAB 1: ALL FUNDS LIST & RANKING
    # ---------------------------------------------------------
    with tab1:
        st.subheader("Comprehensive Fund Performance")
        col1, col2 = st.columns([1, 3])
        with col1:
            sort_options = [
                '5Y Roll Med (%)', '3Y Roll Med (%)', '1Y Roll Med (%)', 
                '1M Return (%)', '1W Return (%)', 'Sharpe Ratio', 'Sortino Ratio', 
                'Max Drawdown (%)', 'Up Capture (%)', 'Down Capture (%)'
            ]
            sort_by = st.selectbox("Rank Funds By:", sort_options)
            asc = True if "Drawdown" in sort_by or "Down Capture" in sort_by else False
            
        with col2:
            st.info(f"Currently ranking by **{sort_by}** ({'Lowest to Highest' if asc else 'Highest to Lowest'})")

        sorted_df = master_df.sort_values(by=sort_by, ascending=asc).reset_index(drop=True)
        numeric_cols = sorted_df.columns.drop('Fund')
        
        st.dataframe(
            sorted_df.style.format("{:.2f}", na_rep="-", subset=numeric_cols)
                           .background_gradient(subset=['1Y Roll Med (%)', '3Y Roll Med (%)', '5Y Roll Med (%)', 'Sharpe Ratio'], cmap='RdYlGn')
                           .background_gradient(subset=['Max Drawdown (%)'], cmap='Reds_r'),
            use_container_width=True, 
            height=600
        )

    # ---------------------------------------------------------
    # TAB 2: SECTOR DEEP DIVE WITH SMART CATEGORIES
    # ---------------------------------------------------------
    with tab2:
        st.markdown("### Segregate Funds by High-Level Category")
        all_categories = sorted(sector_df['Category'].unique().tolist())
        
        selected_categories = st.multiselect(
            "Select Categories (Choose one or multiple):", 
            options=all_categories,
            default=all_categories[0] if all_categories else None
        )
        
        if selected_categories:
            st.markdown(f"#### Analyzing: {', '.join(selected_categories)}")
            
            # Get funds in selected categories
            sec_funds = sector_df[sector_df['Category'].isin(selected_categories)]
            fund_alloc = sec_funds.groupby('Fund')['Allocation (%)'].sum().reset_index()
            fund_names_in_sector = fund_alloc['Fund'].unique().tolist()
            
            if not fund_names_in_sector:
                st.warning("No funds found with allocation to these categories.")
            else:
                # 1. Show all Returns for these specific funds
                st.write("**Returns & Risk Composition for Funds in this Category**")
                cat_metrics = master_df[master_df['Fund'].isin(fund_names_in_sector)]
                num_cols = cat_metrics.columns.drop('Fund')
                st.dataframe(
                    cat_metrics.style.format("{:.2f}", na_rep="-", subset=num_cols)
                               .background_gradient(subset=['1Y Roll Med (%)', '3Y Roll Med (%)'], cmap='RdYlGn'),
                    use_container_width=True
                )
                
                # 2. Sub-selection for Sectoral Comparison
                st.markdown("---")
                st.write("**Compare Sector Allocations of Specific Funds within this Category**")
                sub_select_funds = st.multiselect("Select funds from this category to compare sectors:", fund_names_in_sector)
                
                if sub_select_funds:
                    sub_sec = sector_df[sector_df['Fund'].isin(sub_select_funds)]
                    # Create a pivot table to compare funds side by side
                    pivot_sec = pd.pivot_table(sub_sec, values='Allocation (%)', index='Sector', columns='Fund', aggfunc='sum', fill_value=0)
                    st.dataframe(pivot_sec.style.format("{:.2f}%").background_gradient(cmap='Blues', axis=1), use_container_width=True)

    # ---------------------------------------------------------
    # TAB 3: FUND COMPARISON & DEEP DIVE (NEW)
    # ---------------------------------------------------------
    with tab3:
        st.markdown("### Head-to-Head Fund Comparison")
        all_funds_list = sorted(master_df['Fund'].unique().tolist())
        
        compare_funds = st.multiselect("Search and Select Funds to Compare (e.g., choose 2 or 3 to compare):", all_funds_list)
        
        if compare_funds:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.write("#### 📊 Core Ratios & Metrics")
                comp_metrics = master_df[master_df['Fund'].isin(compare_funds)].set_index('Fund').T
                
                # Format the transposed table nicely
                st.dataframe(comp_metrics.style.format("{:.2f}", na_rep="-"), use_container_width=True, height=450)
            
            with col2:
                st.write("#### 🏢 Sector Overlap")
                comp_sec = sector_df[sector_df['Fund'].isin(compare_funds)]
                pivot_sec = pd.pivot_table(comp_sec, values='Allocation (%)', index='Sector', columns='Fund', aggfunc='sum', fill_value=0)
                # Sort by the first selected fund's allocation for readability
                if len(compare_funds) > 0:
                    pivot_sec = pivot_sec.sort_values(by=compare_funds[0], ascending=False)
                st.dataframe(pivot_sec.style.format("{:.2f}%").background_gradient(cmap='Greens', axis=1), use_container_width=True, height=450)

            st.markdown("---")
            st.write("#### 💼 Top Stock Overlap Comparison")
            comp_stk = stock_df[stock_df['Fund'].isin(compare_funds)]
            pivot_stk = pd.pivot_table(comp_stk, values='Allocation (%)', index=['Company', 'Sector'], columns='Fund', aggfunc='sum', fill_value=0)
            
            # Sort by the average allocation across selected funds to show the most important stocks first
            pivot_stk['Average Alloc'] = pivot_stk.mean(axis=1)
            pivot_stk = pivot_stk.sort_values(by='Average Alloc', ascending=False).drop(columns=['Average Alloc'])
            
            st.dataframe(pivot_stk.head(30).style.format("{:.2f}%").background_gradient(cmap='Purples', axis=1), use_container_width=True)
        else:
            st.info("👆 Please select at least one fund from the dropdown above to begin comparison.")

if __name__ == '__main__':
    main()
