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
    initial_sidebar_state="collapsed"  # 默认隐藏侧边栏
)

# ==========================================
# 移动端友好 CSS
# ==========================================
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* 完全隐藏侧边栏 */
[data-testid="stSidebar"] {
    display: none !important;
}
[data-testid="stSidebarCollapsedControl"] {
    display: none !important;
}

/* 控制面板卡片 */
.control-panel {
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1rem;
}

/* 顶部标题栏 */
.top-bar {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.5rem;
    flex-wrap: wrap;
}

/* 数值显示标签 */
.param-badge {
    display: inline-block;
    background: #1D4ED8;
    color: white;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: bold;
}

/* 移动端按钮更大 */
.run-btn {
    padding: 0.75rem 2rem !important;
    font-size: 1.1rem !important;
}

/* 响应式：小屏幕上指标卡片变两列 */
@media screen and (max-width: 768px) {
    .stColumns {
        flex-wrap: wrap !important;
    }
    h1 {
        font-size: 1.5rem !important;
    }
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# K线图绘制
# ==========================================
def create_mini_kline(df_hist, width_px=240, height_px=80):
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

    for col in ['成交额', '最新价', '涨跌幅', '最高', '最低']:
        if col not in df_spot.columns:
            return None, f"❌ 缺少列: {col}"
        df_spot[col] = pd.to_numeric(df_spot[col], errors='coerce')

    df_spot = df_spot.dropna(subset=['成交额', '最新价'])
    if '名称' in df_spot.columns:
        df_spot = df_spot[~df_spot['名称'].astype(str).str.contains('ST|退|临', na=False)]
    if '代码' in df_spot.columns:
        df_spot['代码'] = df_spot['代码'].astype(str).str.zfill(6)

    df_candidate = df_spot[df_spot['成交额'] > min_volume].copy()
    total = len(df_candidate)
    if df_candidate.empty:
        return None, f"📊 成交额>{min_volume/1e8:.0f}亿 候选为 0"

    status.success(f"📊 初筛候选: {total} 只 | 开始逐只分析历史K线...")

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

            big_idx = hist_10[hist_10['Pct_chg'] > big_yang_threshold].index
            if len(big_idx) == 0:
                continue
            big_i = big_idx[-1]

            big_low = hist_10.loc[big_i, 'Low']
            big_pct = hist_10.loc[big_i, 'Pct_chg']

            after = hist_10.iloc[big_i + 1:]
            if len(after) < min_after_days:
                continue

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
# 初始 Session State
# ==========================================
if 'result_df' not in st.session_state:
    st.session_state.result_df = None
    st.session_state.msg = ""
if 'params' not in st.session_state:
    st.session_state.params = {}
if 'fw' not in st.session_state:
    st.session_state.fw = False

# ==========================================
# 主页面标题
# ==========================================
st.title("🔥 大阳线不破低 · 量化选股")
st.markdown("**策略：** 近10日出现大阳线 → 后续N日不破底 + 成交额稳定 + 振幅可控")

# ==========================================
# 可折叠控制面板（替代侧边栏）
# ==========================================
ctrl = st.expander("⚙️ 参数设置（点击展开/收起）", expanded=not st.session_state.get('hide_ctrl'))

with ctrl:
    # 用两列布局适配手机
    c_left, c_right = st.columns([1, 1])

    with c_left:
        big_yang_pct = st.slider(
            "📈 大阳线最低涨幅(%)", 3.0, 15.0, 5.5, 0.5,
            help="近10日内单日涨幅超过此值视为大阳线"
        )
        min_vol = st.slider(
            "💰 最低成交额(亿元)", 1.0, 10.0, 3.0, 0.5,
            help="候选股当日及后续每日成交额需高于此值"
        )
        min_days = st.slider(
            "📅 后续最少交易日", 2, 10, 3, 1,
            help="大阳线后至少经过多少个交易日"
        )

    with c_right:
        max_amp = st.slider(
            "📊 后续最大振幅(%)", 3.0, 15.0, 8.0, 0.5,
            help="后续每日振幅不得超过此值"
        )
        max_vsv = st.slider(
            "📉 成交额波动率上限", 0.10, 0.60, 0.30, 0.05,
            help="(最大-最小)/最大，超此值则波动过大"
        )

        # 显示当前参数速览
        st.markdown(f"""
        <div style='background:#F3F4F6;border-radius:8px;padding:8px 12px;font-size:0.85rem;margin-top:8px;'>
        当前设定：阳线 <b>{big_yang_pct}%</b> · 成交额 <b>{min_vol}亿</b> · 后续 <b>{min_days}日</b> · 振幅 <b>{max_amp}%</b> · 波动 <b>{max_vsv}</b>
        </div>
        """, unsafe_allow_html=True)

    # 按钮行
    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])
    with btn_col1:
        if st.button("🚀 开始选股", use_container_width=True, type="primary"):
            st.session_state.fw = True
    with btn_col2:
        if st.button("🔄 重置参数", use_container_width=True):
            st.session_state.fw = False
            st.session_state.result_df = None
            st.session_state.msg = ""
            st.rerun()

    st.caption("📡 数据源：新浪财经 / 东方财富实时接口 | 分析周期：近25个交易日")


# ==========================================
# 执行选股
# ==========================================
if st.session_state.fw:
    with st.spinner('⏳ 全市场扫描中，请稍候...（约5000只股票需逐只分析）'):
        result_df, msg = run_screening(
            min_volume=min_vol * 1e8,
            big_yang_threshold=big_yang_pct,
            min_after_days=min_days,
            max_amplitude=max_amp,
            max_volume_volatility=max_vsv,
        )
    st.session_state.result_df = result_df
    st.session_state.msg = msg
    st.session_state.fw = False

# ==========================================
# 结果展示
# ==========================================
if st.session_state.result_df is not None and len(st.session_state.result_df) > 0:
    df = st.session_state.result_df

    st.markdown("---")
    st.success(st.session_state.msg)

    # 指标卡片 - 小屏两列、大屏四列
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🏆 选出", f"{len(df)} 只")
    c2.metric("📈 大阳线均值", f"{df['大阳线涨幅(%)'].mean():.2f}%")
    c3.metric("💰 成交额均值", f"{df['后续成交额均值(亿)'].mean():.2f}亿")
    c4.metric("📊 振幅均值", f"{df['后续最大振幅(%)'].mean():.2f}%")

    # 涨停统计
    limit_count = (df['大阳线涨停'] == '是').sum()
    st.markdown(f"🔴 其中**涨停**大阳线：**{limit_count}** 只 | 非涨停：**{len(df) - limit_count}** 只")

    st.markdown("---")

    # 数据表格
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

    # K线走势图
    st.markdown("### 📊 K线走势（近10日）")
    kline_expander = st.expander("点击展开/收起 K线图", expanded=False)
    with kline_expander:
        for i in range(0, len(df), 2):
            cols_k = st.columns(2)
            for j in range(2):
                idx = i + j
                if idx >= len(df):
                    break
                row = df.iloc[idx]
                with cols_k[j]:
                    try:
                        buf = create_mini_kline(row['历史K线(10日)'], width_px=350, height_px=100)
                        st.image(buf, use_container_width=True)
                    except Exception:
                        st.caption("渲染失败")
                    yz = '🔴涨停' if row['大阳线涨停'] == '是' else '🟢大阳'
                    st.caption(
                        f"**{row['代码']} {row['名称']}** | "
                        f"{yz} {row['大阳线涨幅(%)']:+.2f}% | "
                        f"距底 {row['距大阳线底(%)']:+.2f}%"
                    )

    # 导出
    st.markdown("---")
    cc1, cc2 = st.columns(2)
    with cc1:
        csv = df[display_cols].to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 下载 CSV", csv,
                           f"选股_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                           "text/csv", type="primary")
    with cc2:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as w:
            df[display_cols].to_excel(w, index=False, sheet_name='选股结果')
        st.download_button("📥 下载 Excel", buf.getvalue(),
                           f"选股_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

elif st.session_state.result_df is not None and len(st.session_state.result_df) == 0:
    st.markdown("---")
    st.warning(st.session_state.msg)
    st.info("💡 提示：尝试放宽参数（降低大阳线阈值、放宽振幅限制等）")

else:
    st.markdown("---")
    st.info("👆 在上方控制面板设置参数后，点击「🚀 开始选股」运行策略")
    st.markdown("""
    ### 📋 筛选流程

    | 步骤 | 说明 |
    |------|------|
    | ① | 全市场 A 股约 5000 只，按成交额初筛 |
    | ② | 逐只拉取近 25 日历史 K 线数据 |
    | ③ | 近 10 日内寻找**大阳线**（涨幅 > 设定阈值） |
    | ④ | 验证后续交易日：📌 不破底 · 💰 成交额稳定 · 📊 振幅可控 |
    | ⑤ | ✅ 输出结果 + K 线走势图 + CSV / Excel 导出 |

    > 手机端自动适配，展开/收起均可用 ✅
    """)
