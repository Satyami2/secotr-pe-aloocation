"""
Sector Mutual Fund Analysis Tool - Pro UI
----------------------------------------------
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ---------- File paths ----------
NAV_FILES = ['sector funds 1.xlsx', 'secotr funds 2.xlsx',
             'secotr funds 3.xlsx', 'sector funds 4.xlsx']
SECTOR_ALLOC_FILE = 'secotrs aloocations.xlsx'
STOCK_ALLOC_FILE  = 'sectors stock alocation.xlsx'
PE_FILE           = 'sector pe rstio.xlsx'
SECTOR_CORR_FILE  = 'SECOTRCORR.xlsx'   # NEW

RISK_FREE_RATE = 0.06 

# ============================================================
# DATA LOADING & CATEGORIZATION FUNCTIONS 
# ============================================================
def get_fund_category(fund_name):
    """
    Intelligently categorizes the Fund itself based on the exact lists/images provided.
    If a fund does not match any of these, it defaults to 'Thematic Funds'.
    """
    f = str(fund_name).lower()
    
    # 1. Banks & Finance
    if any(x in f for x in ['banking', 'financial', 'bfsi', 'fin serv']): return 'Bank & Finance'
    # 2. Tech / Teck
    if any(x in f for x in ['tech', 'digital', 'teck']): return 'Tech'
    # 3. Consumption
    if any(x in f for x in ['consum', 'fmcg']): return 'Consumption'
    # 4. Energy & Resources (Checked before Infra due to overlaps)
    if any(x in f for x in ['energy', 'power', 'natural res']): return 'Energy & Resources'
    # 5. Infrastructure
    if any(x in f for x in ['infra', 'build india', 't.i.g.e.r']): return 'Infrastructure'
    # 6. Pharma & Healthcare
    if any(x in f for x in ['pharma', 'health', 'wellness', 'diagnostics']): return 'Pharma & Healthcare'
    # 7. Auto
    if 'auto' in f: return 'Auto'
    # 8. Services (Ensuring we don't accidentally grab Financial Services)
    if 'services' in f and 'financial' not in f and 'fin serv' not in f: return 'Services'
    
    # Fallback for funds not in the main lists
    return 'Thematic Funds'

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
        
        for label, w in windows.items():
            if len(s) > w:
                roll = (s / s.shift(w)) ** (252.0 / w) - 1
                roll = roll.dropna() * 100
                row[f'{label} Roll Med (%)'] = round(roll.median(), 2) if len(roll) else np.nan
            else:
                row[f'{label} Roll Med (%)'] = np.nan

        row['1W Return (%)'] = round((s.iloc[-1] / s.iloc[-6] - 1) * 100, 2) if len(s) >= 6 else np.nan
        row['1M Return (%)'] = round((s.iloc[-1] / s.iloc[-22] - 1) * 100, 2) if len(s) >= 22 else np.nan

        roll_max = s.cummax()
        drawdowns = (s / roll_max - 1) * 100
        row['Max Drawdown (%)'] = round(drawdowns.min(), 2) if len(drawdowns) else np.nan

        fund_up_mean = fund_daily_ret[up_days].mean()
        fund_down_mean = fund_daily_ret[down_days].mean()
        row['Up Capture (%)'] = round((fund_up_mean / bench_up_mean) * 100, 2) if bench_up_mean else np.nan
        row['Down Capture (%)'] = round((fund_down_mean / bench_down_mean) * 100, 2) if bench_down_mean else np.nan

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
            'Avg PE': round(pe.mean(), 2),
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

# ============================================================
# NEW: SECTOR INDEX LOADING (for correlations)
# ============================================================
@st.cache_data
def load_sector_indices():
    """
    Long-format file: Index Name | Date | Close Price
    Returns a wide dataframe: Date index, one column per sector index.
    """
    raw = pd.read_excel(SECTOR_CORR_FILE, header=None,
                        names=['Index', 'Date', 'Close'])
    raw = raw.iloc[3:].copy()                              # skip headers
    raw['Date']  = pd.to_datetime(raw['Date'],  errors='coerce')
    raw['Close'] = pd.to_numeric(raw['Close'], errors='coerce')
    raw = raw.dropna(subset=['Index', 'Date', 'Close'])
    wide = raw.pivot_table(index='Date', columns='Index',
                           values='Close', aggfunc='last')
    wide = wide.sort_index()
    wide.columns.name = None       # avoid 'Index' axis name colliding with reset_index() later
    wide.index.name = 'Date'
    return wide

@st.cache_data
def compute_correlation(indices_df, years):
    """
    Compute correlation matrix of daily returns over the last `years` years.
    years = 0 means use the full history available.
    """
    if years and years > 0:
        cutoff = indices_df.index.max() - pd.DateOffset(years=years)
        df = indices_df.loc[indices_df.index >= cutoff]
    else:
        df = indices_df
    returns = df.pct_change().dropna(how='all')
    # only keep indices with enough data in the window
    returns = returns.dropna(axis=1, thresh=int(len(returns) * 0.5))
    corr = returns.corr()
    return corr

# ============================================================
# MAIN STREAMLIT APP
# ============================================================
def main():
    st.set_page_config(page_title="Pro Fund Analysis", layout="wide")
    st.title("📈 Pro Mutual Fund Analysis & Screener")

    with st.spinner("Calculating metrics & loading data..."):
        nav = load_nav_data()
        metrics_df = compute_advanced_metrics(nav)
        pe_raw = load_pe_data()
        pe_df = pe_summary(pe_raw)
        sector_df = load_sector_allocation()
        stock_df  = load_stock_allocation()
        indices_df = load_sector_indices()              # NEW

        for d in (metrics_df, pe_df, sector_df, stock_df):
            d['Fund'] = d['Fund'].astype(str).str.strip()

    # Create Master Dataframe and assign Categories based on Fund Names
    master_df = metrics_df.merge(pe_df, on='Fund', how='left')
    master_df['Category'] = master_df['Fund'].apply(get_fund_category)

    # ---------- UI Tabs (added a 4th tab for correlations) ----------
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏆 All Funds & Rankings",
        "🗂️ Category Deep Dive",
        "⚖️ Custom Fund Comparison",
        "🔗 Sector Correlations"
    ])

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

        # Organize columns so Category shows up right after Fund
        cols = ['Fund', 'Category'] + [c for c in master_df.columns if c not in ['Fund', 'Category']]
        sorted_df = master_df[cols].sort_values(by=sort_by, ascending=asc).reset_index(drop=True)
        numeric_cols = sorted_df.columns.drop(['Fund', 'Category'])
        
        st.dataframe(
            sorted_df.style.format("{:.2f}", na_rep="-", subset=numeric_cols)
                           .background_gradient(subset=['1Y Roll Med (%)', '3Y Roll Med (%)', '5Y Roll Med (%)', 'Sharpe Ratio'], cmap='RdYlGn')
                           .background_gradient(subset=['Max Drawdown (%)'], cmap='Reds_r'),
            use_container_width=True, 
            height=600
        )

    # ---------------------------------------------------------
    # TAB 2: CATEGORY DEEP DIVE
    # ---------------------------------------------------------
    with tab2:
        st.markdown("### Segregate & Analyze by Category")
        all_categories = sorted(master_df['Category'].unique().tolist())
        
        selected_category = st.selectbox("Select a Category:", all_categories)
        
        if selected_category:
            st.markdown(f"#### 📊 All Funds in {selected_category}")
            
            # Filter master data to only show funds in this specific category
            cat_funds_df = master_df[master_df['Category'] == selected_category].drop(columns=['Category'])
            fund_names_in_category = cat_funds_df['Fund'].tolist()
            
            # Show Returns & Metrics for the Category
            num_cols = cat_funds_df.columns.drop('Fund')
            st.dataframe(
                cat_funds_df.style.format("{:.2f}", na_rep="-", subset=num_cols)
                           .background_gradient(subset=['1Y Roll Med (%)', '3Y Roll Med (%)'], cmap='RdYlGn'),
                use_container_width=True
            )
            
            st.markdown("---")
            st.write("#### 🔍 Compare Composition of specific funds within this category")
            sub_select_funds = st.multiselect(
                f"Select funds from {selected_category} to compare their composition:", 
                fund_names_in_category,
                default=fund_names_in_category[:3] if len(fund_names_in_category) >= 3 else fund_names_in_category
            )
            
            if sub_select_funds:
                colA, colB = st.columns(2)
                with colA:
                    st.write("**Sector Composition Breakdown**")
                    sub_sec = sector_df[sector_df['Fund'].isin(sub_select_funds)]
                    pivot_sec = pd.pivot_table(sub_sec, values='Allocation (%)', index='Sector', columns='Fund', aggfunc='sum', fill_value=0)
                    st.dataframe(pivot_sec.style.format("{:.2f}%").background_gradient(cmap='Blues', axis=1), use_container_width=True)
                
                with colB:
                    st.write("**Top Stock Composition Breakdown**")
                    sub_stk = stock_df[stock_df['Fund'].isin(sub_select_funds)]
                    pivot_stk = pd.pivot_table(sub_stk, values='Allocation (%)', index='Company', columns='Fund', aggfunc='sum', fill_value=0)
                    
                    # Sort stocks by average holding so the biggest ones float to the top
                    pivot_stk['Avg_Holding'] = pivot_stk.mean(axis=1)
                    pivot_stk = pivot_stk.sort_values('Avg_Holding', ascending=False).drop(columns=['Avg_Holding']).head(20)
                    
                    st.dataframe(pivot_stk.style.format("{:.2f}%").background_gradient(cmap='Purples', axis=1), use_container_width=True)

    # ---------------------------------------------------------
    # TAB 3: CUSTOM FUND COMPARISON (Across all categories)
    # ---------------------------------------------------------
    with tab3:
        st.markdown("### Head-to-Head Fund Comparison")
        all_funds_list = sorted(master_df['Fund'].unique().tolist())
        
        compare_funds = st.multiselect("Search and Select Funds to Compare (Pick 2 or more from any category):", all_funds_list)
        
        if compare_funds:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.write("#### 📊 Core Ratios & Metrics")
                # Drop category from transposed view for cleaner look
                comp_metrics = master_df[master_df['Fund'].isin(compare_funds)].drop(columns=['Category']).set_index('Fund').T
                st.dataframe(comp_metrics.style.format("{:.2f}", na_rep="-"), use_container_width=True, height=450)
            
            with col2:
                st.write("#### 🏢 Sector Composition Breakdown")
                comp_sec = sector_df[sector_df['Fund'].isin(compare_funds)]
                pivot_sec = pd.pivot_table(comp_sec, values='Allocation (%)', index='Sector', columns='Fund', aggfunc='sum', fill_value=0)
                if len(compare_funds) > 0:
                    pivot_sec = pivot_sec.sort_values(by=compare_funds[0], ascending=False)
                st.dataframe(pivot_sec.style.format("{:.2f}%").background_gradient(cmap='Greens', axis=1), use_container_width=True, height=450)

            st.markdown("---")
            st.write("#### 💼 Top Stock Composition Breakdown")
            comp_stk = stock_df[stock_df['Fund'].isin(compare_funds)]
            pivot_stk = pd.pivot_table(comp_stk, values='Allocation (%)', index=['Company', 'Sector'], columns='Fund', aggfunc='sum', fill_value=0)
            
            pivot_stk['Average Alloc'] = pivot_stk.mean(axis=1)
            pivot_stk = pivot_stk.sort_values(by='Average Alloc', ascending=False).drop(columns=['Average Alloc'])
            
            st.dataframe(pivot_stk.head(30).style.format("{:.2f}%").background_gradient(cmap='Oranges', axis=1), use_container_width=True)
        else:
            st.info("👆 Please select at least one fund from the dropdown above to begin comparison.")

    # ---------------------------------------------------------
    # TAB 4: SECTOR INDEX CORRELATIONS  (NEW)
    # ---------------------------------------------------------
    with tab4:
        st.markdown("### 🔗 Sector Index Correlations")
        st.caption(
            "Pearson correlation of **daily returns** between Nifty sector indices. "
            "Values close to 1 mean the two sectors move together; close to 0 means "
            "they move independently; negative means they move opposite ways."
        )

        # --- Controls ---
        c1, c2 = st.columns([1, 2])
        with c1:
            window_choice = st.selectbox(
                "Time window:",
                ['1 Year', '3 Years', '5 Years', '10 Years', 'Full History']
            )
        years_map = {'1 Year': 1, '3 Years': 3, '5 Years': 5,
                     '10 Years': 10, 'Full History': 0}
        years = years_map[window_choice]

        corr = compute_correlation(indices_df, years)

        if corr.empty or len(corr) < 2:
            st.warning("Not enough data to compute correlations for this window.")
        else:
            with c2:
                st.info(
                    f"Showing **{len(corr)} sectors** "
                    f"over the **last {years} year(s)**"
                    if years else
                    f"Showing **{len(corr)} sectors** over the **full history**"
                )

            view_mode = st.radio(
                "View:",
                ["Full correlation matrix", "Pick one sector"],
                horizontal=True
            )

            # ---- A) Full matrix ----
            if view_mode == "Full correlation matrix":
                st.write("#### Full Correlation Matrix")
                st.dataframe(
                    corr.style.format("{:.2f}")
                              .background_gradient(cmap='RdYlGn', vmin=-1, vmax=1),
                    use_container_width=True,
                    height=650
                )

                # quick highlights
                # take upper triangle of corr (excluding diagonal) and rank pairs
                tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                pairs = (tri.stack().reset_index()
                            .rename(columns={'level_0': 'Sector A',
                                             'level_1': 'Sector B',
                                             0: 'Correlation'}))
                pairs['Correlation'] = pairs['Correlation'].round(3)

                colA, colB = st.columns(2)
                with colA:
                    st.write("**Most correlated pairs**")
                    st.dataframe(
                        pairs.sort_values('Correlation', ascending=False)
                             .head(10).reset_index(drop=True),
                        use_container_width=True
                    )
                with colB:
                    st.write("**Least correlated pairs (good for diversification)**")
                    st.dataframe(
                        pairs.sort_values('Correlation', ascending=True)
                             .head(10).reset_index(drop=True),
                        use_container_width=True
                    )

            # ---- B) One sector vs all others ----
            else:
                target = st.selectbox(
                    "Choose a sector to see how every other sector correlates with it:",
                    sorted(corr.columns.tolist())
                )

                series = corr[target].drop(target).sort_values(ascending=False)
                table = (series.reset_index()
                              .rename(columns={'index': 'Sector',
                                               target: 'Correlation with ' + target}))
                table['Correlation with ' + target] = (
                    table['Correlation with ' + target].round(3))

                col_l, col_r = st.columns([1, 1])
                with col_l:
                    st.write(f"#### Correlation of every sector with **{target}**")
                    st.dataframe(
                        table.style.format({'Correlation with ' + target: "{:.3f}"})
                                   .background_gradient(
                                       subset=['Correlation with ' + target],
                                       cmap='RdYlGn', vmin=-1, vmax=1),
                        use_container_width=True,
                        height=600
                    )
                with col_r:
                    st.write("**Top 5 most aligned with " + target + "**")
                    st.dataframe(table.head(5).reset_index(drop=True),
                                 use_container_width=True)
                    st.write("**Top 5 least aligned with " + target + "**")
                    st.dataframe(table.tail(5).sort_values(
                                    'Correlation with ' + target).reset_index(drop=True),
                                 use_container_width=True)

                    st.markdown(
                        "💡 *Sectors with **low or negative** correlation with "
                        f"{target} are useful for **diversifying** away from {target}-style risk.*"
                    )

if __name__ == '__main__':
    main()
