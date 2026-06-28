import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import os
import io
import time
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['all_proxy'] = ''
os.environ['ALL_PROXY'] = ''

st.set_page_config(
    page_title="QuantStock | 大阳线不破低选股",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

hide_st_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)


# ==========================================
# K线图绘制
# ==========================================
def create_mini_kline(df_hist, width_px=160, height_px=42):
    fig, ax = plt.subplots(figsize=(width_px / 100, height_px / 100), dpi=100)
    ax.axis('off')
    fig.patch.set_alpha(0)
    ax.set_facecolor('white')

    for i, row in enumerate(df_hist.to_dict('records')):
        open_p, close_p = row['Open'], row['Close']
        high_p, low_p = row['High'], row['Low']
        color = '#EB4335' if close_p >= open_p else '#34A853'
        ax.plot([i, i], [low_p, high_p], color=color, linewidth=1.5)

        box_height = abs(close_p - open_p)
        if box_height == 0:
            box_height = 0.05
        bottom = min(open_p, close_p)
        rect = patches.Rectangle((i - 0.35, bottom), 0.7, box_height,
                                 facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.set_xlim(-1, len(df_hist))
    min_p, max_p = df_hist['Low'].min(), df_hist['High'].max()
    margin = (max_p - min_p) * 0.1
    if margin == 0:
        margin = max_p * 0.01
    ax.set_ylim(min_p - margin, max_p + margin)

    buf = io.BytesIO()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(buf, format='png', transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return buf


# ==========================================
# 选股核心逻辑
# ==========================================
def run_screening(
    min_volume=3e8,
    big_yang_threshold=5.5,
    min_after_days=3,
    max_amplitude=8.0,
    max_volume_volatility=0.30,
    history_days=25
):
    status = st.empty()

    # 获取全市场行情
    status.info("🚀 正在获取全市场实时行情数据...")
    df_spot = None
    for func in ['stock_zh_a_spot_em', 'stock_zh_a_spot']:
        try:
            df_spot = getattr(ak, func)()
            break
        except Exception:
            continue

    if df_spot is None or df_spot.empty:
        return None, "❌ 行情数据获取失败"

    # 类型转换
    for col in ['成交额', '最新价', '涨跌幅', '最高', '最低']:
        if col not in df_spot.columns:
            return None, f"❌ 缺少列: {col}"
        df_spot[col] = pd.to_numeric(df_spot[col], errors='coerce')

    df_spot = df_spot.dropna(subset=['成交额', '最新价'])
    if '名称' in df_spot.columns:
        df_spot = df_spot[~df_spot['名称'].astype(str).str.contains('ST|退|临', na=False)]
    if '代码' in df_spot.columns:
        df_spot['代码'] = df_spot['代码'].astype(str).str.zfill(6)

    # 初筛
    df_candidate = df_spot[df_spot['成交额'] > min_volume].copy()
    total = len(df_candidate)
    if df_candidate.empty:
        return None, f"📊 成交额>{min_volume/1e8:.0f}亿 候选为 0"

    status.success(f"📊 初筛候选: {total} 只 | 开始逐只分析历史K线...")

    # 时间范围
    today = datetime.now()
    end_date = today.strftime('%Y%m%d')
    start_date = (today - timedelta(days=history_days)).strftime('%Y%m%d')

    qualified = []
    progress_bar = st.progress(0, text="准备分析...")

    for idx, (_, row) in enumerate(df_candidate.iterrows(), 1):
        code = row['代码']
        name = row['名称']

        progress_bar.progress(idx / total, text=f"⏳ [{idx}/{total}] {code} {name}")

        try:
            hist = ak.stock_zh_a_hist(symbol=code, period="daily", adjust="qfq",
                                      start_date=start_date, end_date=end_date)
            if hist.empty or len(hist) < 10:
                continue

            hist.rename(columns={
                '开盘': 'Open', '收盘': 'Close', '最高': 'High',
                '最低': 'Low', '成交额': 'Volume', '涨跌幅': 'Pct_chg'
            }, inplace=True)

            hist_10 = hist.tail(10).reset_index(drop=True)

            if 'Pct_chg' not in hist_10.columns:
                hist_10['Pct_chg'] = hist_10['Close'].pct_change() * 100
            hist_10.loc[0, 'Pct_chg'] = 0

            # 大阳线
            big_idx = hist_10[hist_10['Pct_chg'] > big_yang_threshold].index
            if len(big_idx) == 0:
                continue
            big_i = big_idx[-1]

            big_low = hist_10.loc[big_i, 'Low']
            big_pct = hist_10.loc[big_i, 'Pct_chg']

            after = hist_10.iloc[big_i + 1:]
            if len(after) < min_after_days:
                continue

            # 不破底
            if (after['Low'] < big_low).any():
                continue

            after_vols = after['Volume']
            after_amp = (after['High'] - after['Low']) / after['Low'] * 100

            if (after['Volume'] <= min_volume).any():
                continue
            if (after_amp >= max_amplitude).any():
                continue

            vmax, vmin = after_vols.max(), after_vols.min()
            vol_sv = (vmax - vmin) / vmax
            if vol_sv > max_volume_volatility:
                continue

            is_limit = '是' if big_pct >= 9.9 else '否'
            latest = hist_10.iloc[-1]
            by_date = hist_10.iloc[big_i].get('日期', None)

            qualified.append({
                '代码': code,
                '名称': name,
                '最新价': row.get('最新价', np.nan),
                '涨跌幅': row.get('涨跌幅', np.nan),
                '成交额(亿)': row.get('成交额', np.nan) / 1e8 if pd.notna(row.get('成交额')) else np.nan,
                '大阳线日期': by_date,
                '大阳线涨幅(%)': round(big_pct, 2),
                '大阳线最低价': big_low,
                '后续最低价': after['Low'].min(),
                '距大阳线底(%)': round((after['Low'].min() / big_low - 1) * 100, 2),
                '后续成交额均值(亿)': round(after_vols.mean() / 1e8, 2),
                '成交额波动率(%)': round(vol_sv * 100, 2),
                '后续最大振幅(%)': round(after_amp.max(), 2),
                '大阳线涨停': is_limit,
                '历史K线(10日)': hist_10
            })
        except Exception:
            continue

        time.sleep(0.1)

    progress_bar.empty()

    if not qualified:
        return None, "➔ 无符合条件的股票，请放宽参数"

    result_df = pd.DataFrame(qualified)
    result_df = result_df.sort_values('大阳线涨幅(%)', ascending=False)

    return result_df, f"🎯 成功选出 {len(result_df)} 只股票！"


# ==========================================
# 侧边栏
# ==========================================
with st.sidebar:
    st.title("⚙️ 参数设置")
    st.markdown("---")

    big_yang_pct = st.slider("📈 大阳线最低涨幅(%)", 3.0, 15.0, 5.5, 0.5)
    min_vol = st.slider("💰 最低成交额(亿元)", 1.0, 10.0, 3.0, 0.5)
    min_days = st.slider("📅 后续最少交易日", 2, 10, 3, 1)
    max_amp = st.slider("📊 后续最大振幅(%)", 3.0, 15.0, 8.0, 0.5)
    max_vsv = st.slider("📉 成交额波动率上限", 0.10, 0.60, 0.30, 0.05)

    st.markdown("---")

    if st.button("🚀 开始选股", use_container_width=True, type="primary"):
        st.session_state.go = True

    st.caption("数据: 新浪/东方财富 | 缓存: 无")


# ==========================================
# 主页面
# ==========================================
st.title("🔥 大阳线不破低 · 量化选股")
st.markdown("""
**策略：** 近10日出现大阳线 → 后续N日不破底 + 成交额稳定 + 振幅可控
""")

if 'result_df' not in st.session_state:
    st.session_state.result_df = None
    st.session_state.msg = ""

if st.session_state.get('go'):
    with st.spinner('⏳ 全市场扫描中...'):
        result_df, msg = run_screening(
            min_volume=min_vol * 1e8,
            big_yang_threshold=big_yang_pct,
            min_after_days=min_days,
            max_amplitude=max_amp,
            max_volume_volatility=max_vsv,
        )
    st.session_state.result_df = result_df
    st.session_state.msg = msg
    st.session_state.go = False

# 结果展示
if st.session_state.result_df is not None and len(st.session_state.result_df) > 0:
    df = st.session_state.result_df
    st.success(st.session_state.msg)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🏆 选出", f"{len(df)} 只")
    c2.metric("📈 大阳线均值", f"{df['大阳线涨幅(%)'].mean():.2f}%")
    c3.metric("💰 成交额均值", f"{df['后续成交额均值(亿)'].mean():.2f}亿")
    c4.metric("📊 振幅均值", f"{df['后续最大振幅(%)'].mean():.2f}%")

    st.markdown("---")

    display_cols = [
        '代码', '名称', '最新价', '涨跌幅', '成交额(亿)',
        '大阳线日期', '大阳线涨幅(%)', '大阳线最低价', '后续最低价',
        '距大阳线底(%)', '后续成交额均值(亿)', '成交额波动率(%)',
        '后续最大振幅(%)', '大阳线涨停'
    ]

    display_df = df[display_cols].copy()

    def color_pct(val):
        if pd.isna(val):
            return ''
        c = '#DC2626' if val > 0 else '#16A34A'
        return f'color: {c}; font-weight: bold;'

    styled = display_df.style.map(color_pct, subset=['涨跌幅']).format({
        "最新价": "{:.2f}", "涨跌幅": "{:+.2f}%", "成交额(亿)": "{:.2f}",
        "大阳线涨幅(%)": "{:+.2f}%", "大阳线最低价": "{:.2f}", "后续最低价": "{:.2f}",
        "距大阳线底(%)": "{:+.2f}%", "后续成交额均值(亿)": "{:.2f}",
        "成交额波动率(%)": "{:.2f}%", "后续最大振幅(%)": "{:.2f}%"
    })

    st.dataframe(styled, use_container_width=True, hide_index=True, height=500)

    # K线图
    with st.expander("📊 查看 K 线走势图", expanded=False):
        for i in range(0, len(df), 3):
            cols = st.columns(3)
            for j in range(3):
                idx = i + j
                if idx >= len(df):
                    break
                row = df.iloc[idx]
                with cols[j]:
                    try:
                        buf = create_mini_kline(row['历史K线(10日)'], width_px=240, height_px=80)
                        st.image(buf, use_container_width=True)
                    except Exception:
                        st.caption("渲染失败")
                    st.caption(f"**{row['代码']} {row['名称']}** | 大阳线{row['大阳线涨幅(%)']:+.2f}%")

    # 导出
    col_a, col_b = st.columns(2)
    with col_a:
        csv = df[display_cols].to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 下载 CSV", csv,
                           f"选股_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                           "text/csv", type="primary")
    with col_b:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as w:
            df[display_cols].to_excel(w, index=False, sheet_name='选股结果')
        st.download_button("📥 下载 Excel", buf.getvalue(),
                           f"选股_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

elif st.session_state.result_df is not None and len(st.session_state.result_df) == 0:
    st.warning(st.session_state.msg)
    st.info("💡 请放宽左侧参数")

else:
    st.info("👈 请在左侧设置参数后点击「🚀 开始选股」")
    st.markdown("""
    ---
    ### 📋 流程
    ```
    全市场A股 (~5000只)
        │ 成交额初筛
        ▼
    逐只分析近25日历史K线
        │
        ▼
    近10日找大阳线 → 验证后续不破底 + 量能稳定 + 振幅可控
        │
        ▼
    ✅ 输出结果 + K线走势 + CSV/Excel导出
    ```
    """)
