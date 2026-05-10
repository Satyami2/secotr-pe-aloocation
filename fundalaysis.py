"""
Sector Mutual Fund Analysis Tool
--------------------------------
Reads NAV files, PE ratios, sector allocations, and stock allocations.
Produces:
  1. Rolling 1Y / 3Y / 5Y return medians per fund
  2. Harmonic-mean PE per fund (HM is the right average for ratios like P/E)
  3. Sector-wise fund grouping
  4. Sector & stock allocation breakdowns
  5. Combined analysis: pick a sector -> see funds, returns, PE, top holdings
"""

import pandas as pd
import numpy as np
from scipy.stats import hmean
import warnings
warnings.filterwarnings('ignore')

# ---------- File paths ----------
NAV_FILES = ['sector_funds_1.xlsx', 'secotr_funds_2.xlsx',
             'secotr_funds_3.xlsx', 'sector_funds_4.xlsx']
SECTOR_ALLOC_FILE = 'secotrs_aloocations.xlsx'
STOCK_ALLOC_FILE  = 'sectors_stock_alocation.xlsx'
PE_FILE           = 'sector_pe_rstio.xlsx'


# ============================================================
# 1. LOAD NAV DATA
# ============================================================
def load_nav_data():
    """Read all 4 NAV files, return one wide dataframe: Date index, fund columns."""
    all_navs = []
    for f in NAV_FILES:
        raw = pd.read_excel(f, header=None)
        fund_names = raw.iloc[2, 1:].tolist()            # row 2 -> fund names
        data = raw.iloc[4:, :].copy()                     # row 4 onwards -> data
        data.columns = ['Date'] + fund_names
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        data = data.dropna(subset=['Date']).set_index('Date')
        # drop columns that are NaN-named (empty)
        data = data.loc[:, data.columns.notna()]
        data = data.apply(pd.to_numeric, errors='coerce')
        all_navs.append(data)
    nav = pd.concat(all_navs, axis=1, sort=True)
    nav = nav.loc[:, ~nav.columns.duplicated()]          # drop duplicate cols if any
    nav = nav.sort_index()
    return nav


# ============================================================
# 2. ROLLING RETURNS (1Y / 3Y / 5Y) - MEDIAN
# ============================================================
def rolling_returns_summary(nav):
    """
    For every fund compute rolling annualised returns and report the MEDIAN.
    1Y -> 252 trading days, 3Y -> 756, 5Y -> 1260.
    """
    windows = {'1Y': 252, '3Y': 756, '5Y': 1260}
    out = {}
    for fund in nav.columns:
        s = nav[fund].dropna()
        row = {'Fund': fund, 'Data Points': len(s)}
        for label, w in windows.items():
            if len(s) > w:
                # rolling return = (end / start) ^ (1/years) - 1
                roll = (s / s.shift(w)) ** (252.0 / w) - 1
                roll = roll.dropna() * 100
                row[f'{label} Median Return (%)'] = round(roll.median(), 2) if len(roll) else np.nan
            else:
                row[f'{label} Median Return (%)'] = np.nan
        out[fund] = row
    return pd.DataFrame(out).T.reset_index(drop=True)


# ============================================================
# 3. PE RATIOS - HARMONIC MEAN
# ============================================================
def load_pe_data():
    """Parse the PE ratio file. Returns dict { fund_name : DataFrame(date, pe, pbv, dy, mcap) }."""
    raw = pd.read_excel(PE_FILE, header=None)
    funds = {}
    current_fund = None
    rows = []
    for _, r in raw.iterrows():
        cell0 = str(r[0]) if pd.notna(r[0]) else ''
        if cell0.startswith('Scheme Name:'):
            if current_fund and rows:
                funds[current_fund] = pd.DataFrame(
                    rows, columns=['Date', 'PE', 'PBV', 'DY', 'MCAP'])
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
        funds[current_fund] = pd.DataFrame(
            rows, columns=['Date', 'PE', 'PBV', 'DY', 'MCAP'])
    return funds


def pe_summary(pe_data):
    """Harmonic-mean PE per fund (HM is the correct average for ratios).
    Also include latest PE and arithmetic mean for reference."""
    rows = []
    for fund, df in pe_data.items():
        pe = df['PE'].dropna()
        pe = pe[pe > 0]                                  # HM needs positives
        if len(pe) == 0:
            continue
        rows.append({
            'Fund': fund,
            'Latest PE': round(df['PE'].dropna().iloc[0], 2) if df['PE'].dropna().size else np.nan,
            'PE (Harmonic Mean)': round(hmean(pe), 2),
            'PE (Arithmetic Mean)': round(pe.mean(), 2),
            'PE (Median)': round(pe.median(), 2),
            'PBV (HM)': round(hmean(df['PBV'].dropna()[df['PBV'] > 0]), 2)
                         if (df['PBV'].dropna() > 0).any() else np.nan,
            'Months of Data': len(pe),
        })
    return pd.DataFrame(rows)


# ============================================================
# 4. ALLOCATION TABLES
# ============================================================
def load_sector_allocation():
    df = pd.read_excel(SECTOR_ALLOC_FILE, header=3)
    df.columns = ['Fund', 'Sector', 'No of Cos', 'Allocation (%)']
    df = df.dropna(subset=['Fund', 'Sector'])
    df['Allocation (%)'] = pd.to_numeric(df['Allocation (%)'], errors='coerce')
    return df


def load_stock_allocation():
    df = pd.read_excel(STOCK_ALLOC_FILE, header=3)
    df.columns = ['Fund', 'Company', 'Asset', 'Sector', 'Allocation (%)']
    df = df.dropna(subset=['Fund', 'Company'])
    df['Allocation (%)'] = pd.to_numeric(df['Allocation (%)'], errors='coerce')
    return df


# ============================================================
# 5. SECTOR-LEVEL HELPERS
# ============================================================
def funds_by_sector(sector_df, sector_name, top_n=20):
    """Funds with the highest allocation to a chosen sector."""
    s = sector_df[sector_df['Sector'].str.lower() == sector_name.lower()]
    return s.sort_values('Allocation (%)', ascending=False).head(top_n).reset_index(drop=True)


def list_sectors(sector_df):
    return sorted(sector_df['Sector'].dropna().unique().tolist())


# ============================================================
# 6. ONE-FUND DEEP DIVE
# ============================================================
def fund_overview(fund_name, returns_df, pe_df, sector_df, stock_df):
    print(f"\n{'='*70}\n  FUND: {fund_name}\n{'='*70}")

    r = returns_df[returns_df['Fund'].str.strip() == fund_name.strip()]
    if not r.empty:
        print("\nROLLING RETURNS (median, annualised):")
        for c in ['1Y Median Return (%)', '3Y Median Return (%)', '5Y Median Return (%)']:
            v = r.iloc[0][c]
            print(f"  {c:30s} : {v if pd.isna(v) else f'{v:.2f}%'}")
    else:
        print("  No NAV/return data found.")

    p = pe_df[pe_df['Fund'].str.strip() == fund_name.strip()]
    if not p.empty:
        print("\nVALUATION:")
        for c in ['Latest PE', 'PE (Harmonic Mean)', 'PE (Median)', 'PBV (HM)']:
            print(f"  {c:25s} : {p.iloc[0][c]}")

    sec = sector_df[sector_df['Fund'].str.strip() == fund_name.strip()]
    if not sec.empty:
        print("\nSECTOR ALLOCATION:")
        for _, row in sec.sort_values('Allocation (%)', ascending=False).head(8).iterrows():
            print(f"  {row['Sector']:25s} : {row['Allocation (%)']:6.2f}%")

    stk = stock_df[stock_df['Fund'].str.strip() == fund_name.strip()]
    if not stk.empty:
        print("\nTOP STOCK HOLDINGS:")
        for _, row in stk.sort_values('Allocation (%)', ascending=False).head(10).iterrows():
            print(f"  {row['Company']:45s} {row['Sector']:20s} {row['Allocation (%)']:6.2f}%")


# ============================================================
# 7. SECTOR DEEP DIVE -> funds + their PE + their returns
# ============================================================
def sector_deep_dive(sector_name, sector_df, returns_df, pe_df, top_n=15):
    print(f"\n{'='*70}\n  SECTOR ANALYSIS: {sector_name.upper()}\n{'='*70}")

    sub = funds_by_sector(sector_df, sector_name, top_n=top_n)
    if sub.empty:
        print("No funds with allocation in this sector.")
        return None

    # Merge in returns + PE so we get a single comparison table
    sub = sub.merge(returns_df, on='Fund', how='left')
    sub = sub.merge(pe_df[['Fund', 'PE (Harmonic Mean)', 'Latest PE']],
                    on='Fund', how='left')

    cols = ['Fund', 'Allocation (%)',
            '1Y Median Return (%)', '3Y Median Return (%)', '5Y Median Return (%)',
            'Latest PE', 'PE (Harmonic Mean)']
    cols = [c for c in cols if c in sub.columns]

    print(f"\nTop {len(sub)} funds with highest exposure to {sector_name}:\n")
    print(sub[cols].to_string(index=False))

    avg_pe_hm = sub['PE (Harmonic Mean)'].dropna()
    if len(avg_pe_hm):
        print(f"\nAverage PE (HM) across these funds: {hmean(avg_pe_hm[avg_pe_hm > 0]):.2f}")

    return sub


# ============================================================
# MAIN
# ============================================================
def main():
    print("Loading NAV data ...")
    nav = load_nav_data()
    print(f"  -> {nav.shape[1]} funds, {nav.shape[0]} dates "
          f"({nav.index.min().date()} to {nav.index.max().date()})")

    print("\nComputing rolling returns ...")
    returns = rolling_returns_summary(nav)

    print("Loading PE data ...")
    pe_raw = load_pe_data()
    pe_summary_df = pe_summary(pe_raw)
    print(f"  -> PE data for {len(pe_summary_df)} funds")

    print("Loading sector & stock allocations ...")
    sector_df = load_sector_allocation()
    stock_df  = load_stock_allocation()

    # Clean fund names (trim trailing spaces) so merges work
    for d in (returns, pe_summary_df, sector_df, stock_df):
        d['Fund'] = d['Fund'].astype(str).str.strip()

    # ---------- Save consolidated outputs to Excel ----------
    out_path = '/mnt/user-data/outputs/fund_analysis_output.xlsx'
    with pd.ExcelWriter(out_path, engine='openpyxl') as w:
        returns.to_excel(w, sheet_name='Rolling Returns', index=False)
        pe_summary_df.to_excel(w, sheet_name='PE Summary',     index=False)
        sector_df.to_excel(w, sheet_name='Sector Allocation',  index=False)
        stock_df.to_excel(w, sheet_name='Stock Allocation',    index=False)

        # one combined master sheet
        master = (returns
                  .merge(pe_summary_df[['Fund', 'Latest PE',
                                        'PE (Harmonic Mean)', 'PE (Median)']],
                         on='Fund', how='left'))
        master.to_excel(w, sheet_name='Master Summary', index=False)
    print(f"\nSaved consolidated workbook -> {out_path}")

    # ---------- Demo: list sectors ----------
    print("\nAvailable sectors:")
    sectors = list_sectors(sector_df)
    for i, s in enumerate(sectors, 1):
        print(f"  {i:2d}. {s}")

    # ---------- Demo: sector deep dive ----------
    sector_deep_dive('Bank',       sector_df, returns, pe_summary_df, top_n=10)
    sector_deep_dive('IT',         sector_df, returns, pe_summary_df, top_n=10)
    sector_deep_dive('Healthcare', sector_df, returns, pe_summary_df, top_n=10)

    # ---------- Demo: one fund overview ----------
    sample_fund = returns.iloc[0]['Fund']
    fund_overview(sample_fund, returns, pe_summary_df, sector_df, stock_df)

    # ---------- Quick top-list ----------
    print("\n" + "="*70)
    print("  TOP 10 FUNDS BY 5Y MEDIAN ROLLING RETURN")
    print("="*70)
    top5y = (returns.dropna(subset=['5Y Median Return (%)'])
                    .sort_values('5Y Median Return (%)', ascending=False)
                    .head(10))
    print(top5y[['Fund', '1Y Median Return (%)',
                 '3Y Median Return (%)', '5Y Median Return (%)']]
          .to_string(index=False))

    return {
        'nav': nav, 'returns': returns, 'pe': pe_summary_df,
        'sector': sector_df, 'stock': stock_df
    }


if __name__ == '__main__':
    main()
