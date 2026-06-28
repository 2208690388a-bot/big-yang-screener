import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import os
import io
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    initial_sidebar_state="collapsed"
)

# ==========================================
# CSS
# ==========================================
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

[data-testid="stSidebar"] {
    display: none !important;
}
[data-testid="stSidebarCollapsedControl"] {
    display: none !important;
}

@media screen and (max-width: 768px) {
    .stColumns { flex-wrap: wrap !important; }
    h1 { font-size: 1.5rem !important; }
}

/* 登录框居中 */
.login-box {
    max-width: 400px;
    margin: 15vh auto;
    padding: 2rem;
    background: #f8fafc;
    border-radius: 16px;
    border: 1px solid #e2e8f0;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# 板块识别
# ==========================================
def get_board(code_str):
    """根据代码前缀返回板块标签"""
    if code_str.startswith('60'):
        return '沪市主板', '#DC2626'  # 红
    elif code_str.startswith('00'):
        return '深市主板', '#DC2626'  # 红
    elif code_str.startswith('30'):
        return '创业板', '#2563EB'    # 蓝
    elif code_str.startswith('688'):
        return '科创板', '#7C3AED'    # 紫
    elif code_str.startswith('8') or code_str.startswith('4'):
        return '北交所', '#059669'
    return '其他', '#6B7280'


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
# 单只分析（线程池用）
# ==========================================
def analyze_one(code, name, row_data, start_date, end_date,
                big_yang_threshold, min_after_days, min_volume,
                max_amplitude, max_volume_volatility,
                extra_filters_enabled, extra_limits):
    try:
        hist = ak.stock_zh_a_hist(symbol=code, period="daily", adjust="qfq",
                                  start_date=start_date, end_date=end_date)
        if hist.empty or len(hist) < 10:
            return None

        hist.rename(columns={
            '开盘': 'Open', '收盘': 'Close', '最高': 'High',
            '最低': 'Low', '成交额': 'Volume', '涨跌幅': 'Pct_chg'
        }, inplace=True)

        hist_10 = hist.tail(10).reset_index(drop=True)

        if 'Pct_chg' not in hist_10.columns:
            hist_10['Pct_chg'] = hist_10['Close'].pct_change() * 100
        hist_10.loc[0, 'Pct_chg'] = 0

        # ---- ⭐ 额外过滤：市值、流通市值、行业、地区 ----
        if extra_filters_enabled:
            # 总市值过滤
            if extra_limits.get('min_total_cap'):
                total_cap = row_data.get('总市值', np.nan)
                if pd.isna(total_cap) or total_cap < extra_limits['min_total_cap']:
                    return None
            # 流通市值过滤
            if extra_limits.get('min_float_cap'):
                float_cap = row_data.get('流通市值', np.nan)
                if pd.isna(float_cap) or float_cap < extra_limits['min_float_cap']:
                    return None
            # 自由流通市值过滤
            if extra_limits.get('min_free_cap'):
                free_cap = row_data.get('自由流通市值', np.nan)
                if pd.isna(free_cap) or free_cap < extra_limits['min_free_cap']:
                    return None
            # 行业过滤
            if extra_limits.get('industries'):
                industry = row_data.get('所属行业', '')
                if industry and industry not in extra_limits['industries']:
                    return None
            # 地区过滤
            if extra_limits.get('regions'):
                region = row_data.get('所属地区', '')
                if region and region not in extra_limits['regions']:
                    return None

        # ---- 大阳线筛选 ----
        big_idx = hist_10[hist_10['Pct_chg'] > big_yang_threshold].index
        if len(big_idx) == 0:
            return None
        big_i = big_idx[-1]
        big_low = hist_10.loc[big_i, 'Low']
        big_pct = hist_10.loc[big_i, 'Pct_chg']

        after = hist_10.iloc[big_i + 1:]
        if len(after) < min_after_days:
            return None
        if (after['Low'] < big_low).any():
            return None

        after_vols = after['Volume']
        after_amp = (after['High'] - after['Low']) / after['Low'] * 100

        if (after['Volume'] <= min_volume).any():
            return None
        if (after_amp >= max_amplitude).any():
            return None

        vmax, vmin = after_vols.max(), after_vols.min()
        vol_sv = (vmax - vmin) / vmax
        if vol_sv > max_volume_volatility:
            return None

        is_limit = '是' if big_pct >= 9.9 else '否'
        by_date = hist_10.iloc[big_i].get('日期', None)

        # 板块
        board_name, board_color = get_board(code)

        return {
            '代码': code,
            '名称': name,
            '板块': board_name,
            '板块颜色': board_color,
            '最新价': row_data.get('最新价', np.nan),
            '涨跌幅': row_data.get('涨跌幅', np.nan),
            '成交额(亿)': row_data.get('成交额', np.nan) / 1e8 if pd.notna(row_data.get('成交额')) else np.nan,
            '总市值(亿)': row_data.get('总市值', np.nan) / 1e8 if pd.notna(row_data.get('总市值')) else np.nan,
            '流通市值(亿)': row_data.get('流通市值', np.nan) / 1e8 if pd.notna(row_data.get('流通市值')) else np.nan,
            '所属行业': row_data.get('所属行业', ''),
            '所属地区': row_data.get('所属地区', ''),
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
        }
    except Exception:
        return None


# ==========================================
# 批量获取行业/地区/自由流通市值（用于可选过滤）
# ==========================================
@st.cache_data(ttl=3600)
def get_stock_extra_info():
    """返回 code -> {行业, 地区, 自由流通市值} 的映射"""
    mapping = {}
    try:
        # 尝试从东财行业板块获取分类
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                df_ind = ak.stock_board_industry_name_em()
                for _, row in df_ind.iterrows():
                    ind_name = row.get('板块名称', '')
                    try:
                        cons = ak.stock_board_industry_cons_em(symbol=ind_name)
                        for _, s in cons.iterrows():
                            code = str(s.get('代码', ''))
                            if code and len(code) == 6:
                                if code not in mapping:
                                    mapping[code] = {}
                                mapping[code]['所属行业'] = ind_name
                    except Exception:
                        continue
            except Exception:
                pass

        # 尝试获取地区分类
        try:
            df_area = ak.stock_board_area_name_em()
            for _, row in df_area.iterrows():
                area_name = row.get('板块名称', '')
                try:
                    cons = ak.stock_board_area_cons_em(symbol=area_name)
                    for _, s in cons.iterrows():
                        code = str(s.get('代码', ''))
                        if code and len(code) == 6:
                            if code not in mapping:
                                mapping[code] = {}
                            mapping[code]['所属地区'] = area_name
                except Exception:
                    continue
        except Exception:
            pass

        if mapping:
            st.session_state.extra_info_msg = f"✅ 已加载 {len(mapping)} 只股票的行业/地区信息"
        else:
            st.session_state.extra_info_msg = "⚠️ 行业/地区信息加载为空"

    except Exception as e:
        st.session_state.extra_info_msg = f"⚠️ 行业/地区加载失败: {e}"

    return mapping


# ==========================================
# 选股主引擎
# ==========================================
def run_screening(
    min_volume=3e8,
    big_yang_threshold=5.5,
    min_after_days=3,
    max_amplitude=8.0,
    max_volume_volatility=0.30,
    history_days=25,
    extra_filters_enabled=False,
    extra_limits=None
):
    if extra_limits is None:
        extra_limits = {}

    status = st.empty()

    # Step 1: 获取全市场行情
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

    # 额外列转换
    for extra_col in ['总市值', '流通市值']:
        if extra_col in df_spot.columns:
            df_spot[extra_col] = pd.to_numeric(df_spot[extra_col], errors='coerce')

    df_spot = df_spot.dropna(subset=['成交额', '最新价'])
    if '名称' in df_spot.columns:
        df_spot = df_spot[~df_spot['名称'].astype(str).str.contains('ST|退|临', na=False)]
    if '代码' in df_spot.columns:
        df_spot['代码'] = df_spot['代码'].astype(str).str.zfill(6)

    # Step 2: 加载行业/地区映射（仅在启用额外过滤时）
    extra_map = {}
    if extra_filters_enabled:
        need_industry = bool(extra_limits.get('industries'))
        need_region = bool(extra_limits.get('regions'))
        if need_industry or need_region:
            with st.spinner('📋 正在加载行业/地区分类数据...'):
                extra_map = get_stock_extra_info()
                if extra_map:
                    st.success(st.session_state.get('extra_info_msg', '已加载'))
            # 补充到 df_spot
            for idx, row in df_spot.iterrows():
                code = row['代码']
                if code in extra_map:
                    for k, v in extra_map[code].items():
                        if k not in df_spot.columns:
                            df_spot.at[idx, k] = ''
                        df_spot.at[idx, k] = v

    # Step 3: 初筛
    mask = df_spot['成交额'] > min_volume
    if extra_filters_enabled:
        if extra_limits.get('min_total_cap'):
            if '总市值' in df_spot.columns:
                mask = mask & (df_spot['总市值'] >= extra_limits['min_total_cap'])
        if extra_limits.get('min_float_cap'):
            if '流通市值' in df_spot.columns:
                mask = mask & (df_spot['流通市值'] >= extra_limits['min_float_cap'])
    df_candidate = df_spot[mask].copy()
    total = len(df_candidate)

    if df_candidate.empty:
        return None, f"📊 初筛候选为 0（成交额>{min_volume/1e8:.0f}亿），请放宽"

    # Step 4: 日期
    today = datetime.now()
    end_date = today.strftime('%Y%m%d')
    start_date = (today - timedelta(days=history_days)).strftime('%Y%m%d')

    workers = min(16, max(8, total // 20))
    status.success(
        f"📊 初筛候选: {total} 只 | "
        f"⚡ 并发 {workers} 线程 | "
        f"预计 {total/(workers*3):.0f}-{total/(workers*2):.0f} 分钟"
    )

    # Step 5: 多线程分析
    qualified = []
    progress_bar = st.progress(0, text=f"⚡ 并发 {workers} 线程启动中...")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for _, row in df_candidate.iterrows():
            code = row['代码']
            name = row['名称']
            f = executor.submit(
                analyze_one, code, name, row,
                start_date, end_date,
                big_yang_threshold, min_after_days, min_volume,
                max_amplitude, max_volume_volatility,
                extra_filters_enabled, extra_limits
            )
            futures[f] = code

        completed = 0
        for future in as_completed(futures):
            completed += 1
            code = futures[future]
            pct = completed / total
            progress_bar.progress(pct,
                text=f"⚡ [{completed}/{total}] {code} | 并发{workers}线程 | {pct*100:.0f}%")
            result = future.result()
            if result is not None:
                qualified.append(result)

    progress_bar.empty()

    if not qualified:
        return None, "➔ 无符合条件的股票，请放宽参数"

    result_df = pd.DataFrame(qualified)
    result_df = result_df.sort_values('大阳线涨幅(%)', ascending=False)

    return result_df, f"🎯 选出 {len(result_df)} 只（并发 {workers} 线程 / 候选 {total} 只）"


# ==========================================
# 密码验证
# ==========================================
PASSWORD = "040528"

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown("""
    <div class="login-box">
        <h2>🔐 量化选股系统</h2>
        <p style="color:#6B7280;">请输入密码以继续</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        pwd = st.text_input("密码", type="password", placeholder="请输入访问密码",
                           key="pwd_input", label_visibility="collapsed")
        if st.button("🔓 登录", use_container_width=True, type="primary"):
            if pwd == PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("❌ 密码错误")

    st.stop()


# ==========================================
# 初始化 Session State
# ==========================================
for key, default in [('result_df', None), ('msg', ''), ('fw', False),
                      ('extra_info_msg', ''), ('industry_data_loaded', False)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ==========================================
# 主页面
# ==========================================
st.title("🔥 大阳线不破低 · 量化选股")
st.markdown("**策略：** 近10日出现大阳线 → 后续N日不破底 + 成交额稳定 + 振幅可控")

# ==========================================
# 控制面板
# ==========================================
ctrl = st.expander("⚙️ 参数设置（点击展开/收起）", expanded=True)

with ctrl:
    c_left, c_right = st.columns([1, 1])

    with c_left:
        big_yang_pct = st.slider(
            "📈 大阳线最低涨幅(%)", 3.0, 15.0, 5.5, 0.5)
        min_vol = st.slider(
            "💰 最低成交额(亿元)", 1.0, 10.0, 3.0, 0.5)
        min_days = st.slider(
            "📅 后续最少交易日", 2, 10, 3, 1)

    with c_right:
        max_amp = st.slider(
            "📊 后续最大振幅(%)", 3.0, 15.0, 8.0, 0.5)
        max_vsv = st.slider(
            "📉 成交额波动率上限", 0.10, 0.60, 0.30, 0.05)

        st.markdown(f"""
        <div style='background:#F3F4F6;border-radius:8px;padding:8px 12px;font-size:0.85rem;margin-top:8px;'>
        阳线 <b>{big_yang_pct}%</b> · 成交额 <b>{min_vol}亿</b> · 后续 <b>{min_days}日</b> · 振幅 <b>{max_amp}%</b> · 波动 <b>{max_vsv}</b>
        </div>
        """, unsafe_allow_html=True)

    # ---- ⭐ 可选额外过滤 ----
    st.markdown("---")
    st.markdown("### 🔧 可选筛选条件（按需启用）")

    # 默认全部折叠
    use_extra = st.checkbox("✅ 启用自定义筛选条件", value=False,
                            help="勾选后可设定市值/流通市值/行业/地区等额外过滤条件")

    extra_limits = {}
    extra_filters_enabled = False

    if use_extra:
        extra_filters_enabled = True
        ex1, ex2 = st.columns([1, 1])

        with ex1:
            use_total_cap = st.checkbox("🏢 总市值过滤", value=False)
            if use_total_cap:
                min_total_cap = st.number_input(
                    "最小总市值(亿元)", min_value=0.0, value=50.0, step=10.0,
                    help="仅保留总市值大于此值的股票")
                extra_limits['min_total_cap'] = min_total_cap * 1e8

            use_free_cap = st.checkbox("💎 自由流通市值过滤", value=False)
            if use_free_cap:
                min_free_cap = st.number_input(
                    "最小自由流通市值(亿元)", min_value=0.0, value=10.0, step=5.0,
                    help="自由流通市值大致=流通市值-大股东/战略投资者持股")
                extra_limits['min_free_cap'] = min_free_cap * 1e8

            use_industry = st.checkbox("🏭 所属行业过滤", value=False)
            if use_industry:
                selected_industries = None
                # 延迟加载行业列表
                if not st.session_state.get('industry_list'):
                    try:
                        df_ind_names = ak.stock_board_industry_name_em()
                        st.session_state.industry_list = sorted(
                            df_ind_names['板块名称'].dropna().unique().tolist())
                    except Exception:
                        st.session_state.industry_list = []
                        st.warning("行业列表加载失败")

                if st.session_state.get('industry_list'):
                    selected_industries = st.multiselect(
                        "选择行业（可多选，留空=全部）",
                        options=st.session_state.industry_list,
                        default=[],
                        placeholder="例如: 半导体、白酒、新能源...")
                    if selected_industries:
                        extra_limits['industries'] = selected_industries

        with ex2:
            use_float_cap = st.checkbox("📦 流通市值过滤", value=False)
            if use_float_cap:
                min_float_cap = st.number_input(
                    "最小流通市值(亿元)", min_value=0.0, value=20.0, step=5.0,
                    help="仅保留流通市值大于此值的股票")
                extra_limits['min_float_cap'] = min_float_cap * 1e8

            use_region = st.checkbox("📍 所属地区过滤", value=False)
            if use_region:
                selected_regions = None
                if not st.session_state.get('region_list'):
                    try:
                        df_area_names = ak.stock_board_area_name_em()
                        st.session_state.region_list = sorted(
                            df_area_names['板块名称'].dropna().unique().tolist())
                    except Exception:
                        st.session_state.region_list = []
                        st.warning("地区列表加载失败")

                if st.session_state.get('region_list'):
                    selected_regions = st.multiselect(
                        "选择地区（可多选，留空=全部）",
                        options=st.session_state.region_list,
                        default=[],
                        placeholder="例如: 上海、广东、北京...")
                    if selected_regions:
                        extra_limits['regions'] = selected_regions

        # 额外筛选摘要
        active_filters = []
        if extra_limits.get('min_total_cap'):
            active_filters.append(f"总市值>{extra_limits['min_total_cap']/1e8:.0f}亿")
        if extra_limits.get('min_float_cap'):
            active_filters.append(f"流通市值>{extra_limits['min_float_cap']/1e8:.0f}亿")
        if extra_limits.get('min_free_cap'):
            active_filters.append(f"自由流通>{extra_limits['min_free_cap']/1e8:.0f}亿")
        if extra_limits.get('industries'):
            active_filters.append(f"行业:{','.join(extra_limits['industries'][:3])}{'...' if len(extra_limits['industries'])>3 else ''}")
        if extra_limits.get('regions'):
            active_filters.append(f"地区:{','.join(extra_limits['regions'][:3])}{'...' if len(extra_limits['regions'])>3 else ''}")

        if active_filters:
            st.info(f"📌 已启用额外筛选：{' | '.join(active_filters)}")
        else:
            st.caption("💡 勾选上方选项以启用对应的附加筛选条件")

    st.markdown("---")

    # 按钮
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
    with btn_col3:
        if st.button("🚪 退出登录", use_container_width=True):
            st.session_state.authenticated = False
            st.rerun()

    st.caption(f"📡 数据：新浪/东财 | 周期：近25日 | 板块标注：🔴主板 🔵创业板 🟣科创板")


# ==========================================
# 执行选股
# ==========================================
if st.session_state.fw:
    with st.spinner('⏳ 全市场扫描中...'):
        result_df, msg = run_screening(
            min_volume=min_vol * 1e8,
            big_yang_threshold=big_yang_pct,
            min_after_days=min_days,
            max_amplitude=max_amp,
            max_volume_volatility=max_vsv,
            extra_filters_enabled=extra_filters_enabled,
            extra_limits=extra_limits
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

    # 板块分布
    board_counts = df['板块'].value_counts()
    board_str = ' | '.join([f"{b} {c}只" for b, c in board_counts.items()])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🏆 选出", f"{len(df)} 只", delta=board_str[:50] if board_str else None)
    c2.metric("📈 大阳线均值", f"{df['大阳线涨幅(%)'].mean():.2f}%")
    c3.metric("💰 成交额均值", f"{df['后续成交额均值(亿)'].mean():.2f}亿")
    c4.metric("📊 振幅均值", f"{df['后续最大振幅(%)'].mean():.2f}%")

    limit_count = (df['大阳线涨停'] == '是').sum()
    st.markdown(
        f"🔴 涨停大阳：**{limit_count}** 只 | "
        f"🟢 非涨停：**{len(df)-limit_count}** 只 | "
        f"🏷️ 板块分布：{board_str}"
    )

    st.markdown("---")

    # ---- 带板块颜色的表格 ----
    display_cols = [
        '代码', '名称', '板块', '最新价', '涨跌幅', '成交额(亿)',
        '总市值(亿)', '流通市值(亿)', '所属行业', '所属地区',
        '大阳线日期', '大阳线涨幅(%)', '大阳线最低价', '后续最低价',
        '距大阳线底(%)', '后续成交额均值(亿)', '成交额波动率(%)',
        '后续最大振幅(%)', '大阳线涨停'
    ]

    display_df = df[display_cols].copy()

    def color_name_by_board(row):
        """根据板块给名称列着色"""
        try:
            code = str(row['代码'])
            _, color = get_board(code)
            return [f'color: {color}; font-weight: bold;'] * len(row)
        except Exception:
            return [''] * len(row)

    def color_pct(val):
        if pd.isna(val):
            return ''
        c = '#DC2626' if val > 0 else '#16A34A'
        return f'color: {c}; font-weight: bold;'

    # 先整体格式化
    fmt_dict = {
        "最新价": "{:.2f}", "涨跌幅": "{:+.2f}%", "成交额(亿)": "{:.2f}",
        "总市值(亿)": "{:.2f}", "流通市值(亿)": "{:.2f}",
        "大阳线涨幅(%)": "{:+.2f}%", "大阳线最低价": "{:.2f}", "后续最低价": "{:.2f}",
        "距大阳线底(%)": "{:+.2f}%", "后续成交额均值(亿)": "{:.2f}",
        "成交额波动率(%)": "{:.2f}%", "后续最大振幅(%)": "{:.2f}%"
    }

    styled = display_df.style \
        .map(color_pct, subset=['涨跌幅']) \
        .format(fmt_dict)

    st.dataframe(styled, use_container_width=True, hide_index=True, height=500)

    # 板块图例
    st.markdown("🔴 **沪深主板** · 🔵 **创业板** · 🟣 **科创板** · 🟢 **北交所**")

    # K线图
    st.markdown("### 📊 K线走势（近10日）")
    with st.expander("点击展开/收起 K线图", expanded=False):
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
                    bd = row.get('板块', '')
                    st.caption(
                        f"**{row['代码']} {row['名称']}** [{bd}] | "
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
    st.info("💡 提示：尝试放宽参数")

else:
    st.markdown("---")
    st.info("👆 设置参数后点击「🚀 开始选股」")
    st.markdown("""
    ### 📋 流程
    | 步骤 | 说明 |
    |------|------|
    | ① | 全市场 A 股约 5000 只，按成交额初筛 |
    | ② | 多线程并发拉取近 25 日历史 K 线 |
    | ③ | 近 10 日内找大阳线 + 验证不破底 / 量稳 / 振幅可控 |
    | ④ | 可选额外过滤：市值 · 流通市值 · 行业 · 地区 |
    | ⑤ | ✅ 输出结果 + K 线图 + CSV/Excel 导出 |

    > 🔴沪深主板红 · 🔵创业板蓝 · 🟣科创板紫 · 🟢北交所绿
    """)
