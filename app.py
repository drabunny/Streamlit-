import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.preprocessing import LabelEncoder
from io import BytesIO

# ================== 页面配置 ==================
st.set_page_config(
    page_title="房价预测与影响因素分析系统",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================== CSS ==================
st.markdown("""
<style>
    .stApp {
        background-color: #f5f7fa;
    }
    .main > div {
        background-color: #ffffff;
        border-radius: 20px;
        padding: 2.5rem;
        margin: 1rem;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.05);
    }
    .stButton > button {
        background: linear-gradient(135deg, #1677ff 0%, #4096ff 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(22, 119, 255, 0.15);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #0958d9 0%, #1677ff 100%);
        box-shadow: 0 4px 12px rgba(22, 119, 255, 0.25);
        transform: translateY(-2px);
    }
    .result-card {
        background-color: #ffffff;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
        margin-bottom: 1.5rem;
        border-left: 6px solid #1677ff;
        text-align: center;
    }
    .metric-value {
        font-size: 3rem;
        font-weight: 800;
        color: #1677ff;
        line-height: 1.1;
    }
    .metric-unit {
        font-size: 1.25rem;
        color: #4e5969;
        font-weight: 500;
    }
    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1d2129;
        margin: 1rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #f0f0f0;
    }
    .stExpander {
        border-radius: 12px !important;
        border: 1px solid #f0f0f0;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
        margin-bottom: 1rem;
        overflow: hidden;
    }
    .info-card {
        background-color: #f7f8fa;
        padding: 1.25rem;
        border-radius: 12px;
        border-left: 4px solid #1677ff;
        margin-top: 1rem;
    }
    hr {
        border-color: #f0f0f0;
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ================== 加载模型与处理对象 ==================
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_xgboost_tuned.pkl")
    feature_cols = joblib.load("feature_columns.pkl")
    encoders = joblib.load("label_encoders.pkl")
    y_mean = np.load("y_train_mean.npy").item()
    return model, feature_cols, encoders, y_mean

try:
    model, FEATURE_COLS, encoders, y_train_mean = load_artifacts()
    train_rmse = 2988.51
    train_mape_percent = (train_rmse / y_train_mean) * 100
except FileNotFoundError as e:
    st.error(f"❌ 缺少必要的模型文件：{e}")
    st.stop()

# ================== 中文字体 ==================
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

# ================== 宏观数据字典（城市+年份）==================
MACRO_DATA = {
    ("济南市", 2021): {"income": 57449, "gdp": 122400, "population": 933.6, "tertiary": 61.7},
    ("济南市", 2022): {"income": 59459, "gdp": 127800, "population": 941.5, "tertiary": 61.7},
    ("济南市", 2023): {"income": 62506, "gdp": 135200, "population": 943.7, "tertiary": 62.8},
    ("济南市", 2024): {"income": 65364, "gdp": 142200, "population": 951.5, "tertiary": 63.3},
    ("烟台市", 2021): {"income": 53169, "gdp": 122818, "population": 708.28, "tertiary": 51.5},
    ("烟台市", 2022): {"income": 55700, "gdp": 134581, "population": 705.87, "tertiary": 50.8},
    ("烟台市", 2023): {"income": 59126, "gdp": 144241, "population": 703.22, "tertiary": 51.0},
    ("烟台市", 2024): {"income": 62060, "gdp": 153300, "population": 703.52, "tertiary": 51.1},
    ("济宁市", 2021): {"income": 41256, "gdp": 60728, "population": 833.7, "tertiary": 48.4},
    ("济宁市", 2022): {"income": 42989, "gdp": 64100, "population": 829.06, "tertiary": 49.6},
    ("济宁市", 2023): {"income": 45055, "gdp": 66741, "population": 824.05, "tertiary": 51.03},
    ("济宁市", 2024): {"income": 47812, "gdp": 71600, "population": 818.73, "tertiary": 52.07},
}

# ================== 辅助函数 ==================
def encode_categorical(value, encoder):
    try:
        return encoder.transform([value])[0]
    except ValueError:
        if '其他' in encoder.classes_:
            return encoder.transform(['其他'])[0]
        else:
            return encoder.transform([encoder.classes_[0]])[0]

def predict_price(input_dict):
    input_df = pd.DataFrame([input_dict])[FEATURE_COLS]
    return model.predict(input_df)[0]

def plot_shap_waterfall_clean(input_dict):
    input_df = pd.DataFrame([input_dict])[FEATURE_COLS]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    expected_value = explainer.expected_value
    # 计算最终预测值
    final_pred = predict_price(input_dict)

    # 创建图形
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
    # 绘制瀑布图
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=expected_value,
            data=input_df.iloc[0].values,
            feature_names=FEATURE_COLS
        ),
        show=False,
        max_display=15,
        ax=ax
    )
    # 隐藏多余的标题和数值
    for text in ax.texts:
        if 'model output value' in text.get_text().lower() or 'base value' in text.get_text().lower():
            text.set_visible(False)
    # 在合适位置添加统一的预测值标签
    ax.set_title(f'房价影响因素瀑布图\n最终预测值: {final_pred:.0f} 元/平米', fontsize=14, pad=20)
    ax.annotate(f'预测值 = {final_pred:.0f}', xy=(0.95, 0.05), xycoords='axes fraction',
                fontsize=12, ha='right', color='#1677ff', weight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#1677ff"))
    plt.tight_layout()
    return fig

# ================== 初始化 session_state ==================
if 'init_done' not in st.session_state:
    st.session_state.area = 100.0
    st.session_state.age = 5
    st.session_state.orientation = '南'
    st.session_state.decoration = '精装'
    st.session_state.elevator = '有'
    st.session_state.dist_subway = 1500
    st.session_state.count_subway = 2
    st.session_state.dist_bus = 500
    st.session_state.count_bus = 10
    st.session_state.dist_school = 1000
    st.session_state.count_school = 5
    st.session_state.dist_hospital = 2000
    st.session_state.count_hospital = 2
    st.session_state.dist_clinic = 800
    st.session_state.count_clinic = 4
    st.session_state.dist_pharmacy = 300
    st.session_state.count_pharmacy = 8
    st.session_state.dist_mall = 1500
    st.session_state.count_mall = 1
    st.session_state.dist_small_business = 200
    st.session_state.count_small_business = 20
    st.session_state.dist_catering = 100
    st.session_state.count_catering = 30
    st.session_state.dist_park = 1000
    st.session_state.count_park = 2
    # 默认城市和年份
    st.session_state.city = "济南市"
    st.session_state.year = 2023
    # 初始宏观数据
    default_macro = MACRO_DATA.get((st.session_state.city, st.session_state.year), MACRO_DATA[("济南市", 2023)])
    st.session_state.income = default_macro["income"]
    st.session_state.gdp = default_macro["gdp"]
    st.session_state.population = default_macro["population"]
    st.session_state.tertiary = default_macro["tertiary"]
    st.session_state.init_done = True

# ================== 页面主标题 ==================
st.markdown(
    "<h1 style='text-align: center; color: #1677ff; margin-bottom: 2rem; font-weight: 800;'>🏠 房价预测与影响因素分析系统</h1>",
    unsafe_allow_html=True
)

# ================== 宏观经济指标（带城市年份选择）==================
st.markdown("<div class='section-title'>📊 城市宏观经济指标</div>", unsafe_allow_html=True)

# 城市和年份选择
col_city, col_year, _ = st.columns([1, 1, 2])
with col_city:
    selected_city = st.selectbox("选择城市", ["济南市", "烟台市", "济宁市"], key="city_select")
with col_year:
    selected_year = st.selectbox("选择年份", [2021, 2022, 2023, 2024], key="year_select")

# 根据选择更新宏观数据（如果发生变化）
if selected_city != st.session_state.city or selected_year != st.session_state.year:
    st.session_state.city = selected_city
    st.session_state.year = selected_year
    macro_vals = MACRO_DATA.get((selected_city, selected_year), MACRO_DATA[("济南市", 2023)])
    st.session_state.income = macro_vals["income"]
    st.session_state.gdp = macro_vals["gdp"]
    st.session_state.population = macro_vals["population"]
    st.session_state.tertiary = macro_vals["tertiary"]

# 显示可编辑的宏观指标
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.number_input("城镇居民人均可支配收入 (元)", key="income", min_value=30000, max_value=100000, step=1000)
with col_m2:
    st.number_input("人均GDP (元)", key="gdp", min_value=50000, max_value=200000, step=5000)
with col_m3:
    st.number_input("常住人口 (万人)", key="population", min_value=500.0, max_value=1200.0, step=10.0)
with col_m4:
    st.number_input("第三产业占比 (%)", key="tertiary", min_value=40.0, max_value=80.0, step=1.0)

st.divider()

# ================== 左右主布局 ==================
col_left, col_right = st.columns([1, 1.2], gap="large")

with col_left:
    st.markdown("<div class='section-title'>🏷️ 房屋基础信息</div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.number_input("建筑面积 (㎡)", key="area", min_value=30.0, max_value=300.0, step=1.0)
    with col2:
        st.number_input("房龄 (年)", key="age", min_value=0, max_value=50, step=1)
    with col3:
        st.selectbox("房屋朝向", ["南", "北", "东", "西", "其他"], key="orientation")

    col4, col5 = st.columns(2)
    with col4:
        st.selectbox("装修程度", ["精装", "简装", "毛坯", "其他", "未知"], key="decoration")
    with col5:
        st.selectbox("电梯配置", ["有", "无", "未知"], key="elevator")

    st.markdown("<div class='section-title'>🚏 周边配套设施</div>", unsafe_allow_html=True)

    with st.expander("🚇 交通设施", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.number_input("距最近地铁站 (米)", key="dist_subway", min_value=0, max_value=20000, step=100)
        with c2:
            st.number_input("10km内地铁站数量", key="count_subway", min_value=0, max_value=20, step=1)
        c1, c2 = st.columns(2)
        with c1:
            st.number_input("距最近公交站 (米)", key="dist_bus", min_value=0, max_value=5000, step=100)
        with c2:
            st.number_input("10km内公交站数量", key="count_bus", min_value=0, max_value=100, step=5)

    with st.expander("🏥 教育医疗"):
        c1, c2 = st.columns(2)
        with c1:
            st.number_input("距最近学校 (米)", key="dist_school", min_value=0, max_value=10000, step=100)
        with c2:
            st.number_input("10km内学校数量", key="count_school", min_value=0, max_value=50, step=1)
        c1, c2 = st.columns(2)
        with c1:
            st.number_input("距最近综合医院 (米)", key="dist_hospital", min_value=0, max_value=15000, step=100)
        with c2:
            st.number_input("10km内综合医院数量", key="count_hospital", min_value=0, max_value=20, step=1)
        c1, c2 = st.columns(2)
        with c1:
            st.number_input("距最近诊所 (米)", key="dist_clinic", min_value=0, max_value=5000, step=100)
        with c2:
            st.number_input("10km内诊所数量", key="count_clinic", min_value=0, max_value=50, step=1)
        c1, c2 = st.columns(2)
        with c1:
            st.number_input("距最近药店 (米)", key="dist_pharmacy", min_value=0, max_value=2000, step=100)
        with c2:
            st.number_input("10km内药店数量", key="count_pharmacy", min_value=0, max_value=100, step=1)

    with st.expander("🛍️ 商业休闲"):
        c1, c2 = st.columns(2)
        with c1:
            st.number_input("距最近大型商场 (米)", key="dist_mall", min_value=0, max_value=10000, step=100)
        with c2:
            st.number_input("10km内大型商场数量", key="count_mall", min_value=0, max_value=20, step=1)
        c1, c2 = st.columns(2)
        with c1:
            st.number_input("距最近小型商业 (米)", key="dist_small_business", min_value=0, max_value=3000, step=100)
        with c2:
            st.number_input("10km内小型商业数量", key="count_small_business", min_value=0, max_value=200, step=5)
        c1, c2 = st.columns(2)
        with c1:
            st.number_input("距最近餐饮场所 (米)", key="dist_catering", min_value=0, max_value=2000, step=50)
        with c2:
            st.number_input("10km内餐饮数量", key="count_catering", min_value=0, max_value=300, step=10)
        c1, c2 = st.columns(2)
        with c1:
            st.number_input("距最近公园 (米)", key="dist_park", min_value=0, max_value=10000, step=100)
        with c2:
            st.number_input("10km内公园数量", key="count_park", min_value=0, max_value=20, step=1)

with col_right:
    st.markdown("<div class='section-title'>📈 房价预测结果</div>", unsafe_allow_html=True)
    predict_btn = st.button("🔮 开始预测房价", type="primary", use_container_width=True)

    if predict_btn:
        try:
            input_dict = {
                '建筑面积': st.session_state.area,
                '房龄': st.session_state.age,
                '朝向': encode_categorical(st.session_state.orientation, encoders['朝向']),
                '装修': encode_categorical(st.session_state.decoration, encoders['装修']),
                '有无电梯': encode_categorical(st.session_state.elevator, encoders['有无电梯']),
                'dist_地铁站': st.session_state.dist_subway,
                'count_地铁站_within_10000m': st.session_state.count_subway,
                'dist_公交站': st.session_state.dist_bus,
                'count_公交站_within_10000m': st.session_state.count_bus,
                'dist_学校': st.session_state.dist_school,
                'count_学校_within_10000m': st.session_state.count_school,
                'dist_综合医院': st.session_state.dist_hospital,
                'count_综合医院_within_10000m': st.session_state.count_hospital,
                'dist_诊所/社区医院': st.session_state.dist_clinic,
                'count_诊所/社区医院_within_10000m': st.session_state.count_clinic,
                'dist_药店': st.session_state.dist_pharmacy,
                'count_药店_within_10000m': st.session_state.count_pharmacy,
                'dist_大型商场': st.session_state.dist_mall,
                'count_大型商场_within_10000m': st.session_state.count_mall,
                'dist_小型商业': st.session_state.dist_small_business,
                'count_小型商业_within_10000m': st.session_state.count_small_business,
                'dist_餐饮': st.session_state.dist_catering,
                'count_餐饮_within_10000m': st.session_state.count_catering,
                'dist_公园': st.session_state.dist_park,
                'count_公园_within_10000m': st.session_state.count_park,
                '城镇居民人均可支配收入': st.session_state.income,
                '人均GDP': st.session_state.gdp,
                '常住人口': st.session_state.population,
                '第三产业占比': st.session_state.tertiary,
            }
            with st.spinner("模型计算中，请稍候..."):
                pred = predict_price(input_dict)
                fig = plot_shap_waterfall_clean(input_dict)
            st.session_state['pred'] = pred
            st.session_state['fig'] = fig
        except Exception as e:
            st.error(f"预测失败: {str(e)}")

    if 'pred' in st.session_state:
        pred = st.session_state.pred
        st.markdown(f'''
        <div class="result-card">
            <div class="metric-value">{pred:.0f}</div>
            <div class="metric-unit">元/平方米</div>
        </div>
        ''', unsafe_allow_html=True)

        st.caption(f"📊 训练集基准均价：{y_train_mean:.0f} 元/平米")
        st.caption(f"⚙️ 模型误差 RMSE：{train_rmse:.0f} ｜ 相对误差：{train_mape_percent:.1f}%")

        st.markdown("<div class='section-title'>🔍 房价影响因素深度分析</div>", unsafe_allow_html=True)
        st.pyplot(st.session_state.fig, use_container_width=True)

        buf = BytesIO()
        st.session_state.fig.savefig(buf, format="png", dpi=180, bbox_inches='tight')
        buf.seek(0)
        st.download_button(
            label="📥 高清下载分析图",
            data=buf,
            file_name="房价影响因素_高清分析图.png",
            mime="image/png",
            use_container_width=True
        )
    else:
        st.markdown("""
        <div class="info-card">
            <h4 style='margin-top:0; color:#1677ff;'>💡 操作指南</h4>
            <p style='margin-bottom:0; line-height:1.7;'>
            1. 选择城市和年份，自动填充宏观数据，也可手动修改<br>
            2. 填写房屋基础属性、周边配套参数<br>
            3. 点击【开始预测房价】按钮一键计算<br>
            4. 自动生成因素分解图，直观查看涨跌原因
            </p>
        </div>
        """, unsafe_allow_html=True)

# ================== 页脚 ==================
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #86909c;'>© 2025 房价预测系统 | XGBoost+SHAP | 数据范围：山东济南/烟台/济宁 2021-2024</p>",
    unsafe_allow_html=True
)
