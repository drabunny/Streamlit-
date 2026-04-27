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

# ================== 自定义CSS样式（美化版） ==================
st.markdown("""
<style>
    /* 全局样式 */
    .main {
        background-color: #f8fafc;
    }
    
    /* 按钮样式 */
    .stButton > button {
        background: linear-gradient(90deg, #0066cc, #0099ff);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: bold;
        font-size: 1rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #0055aa, #0088ee);
        box-shadow: 0 6px 16px rgba(0,102,204,0.25);
        transform: translateY(-2px);
    }
    
    /* 结果卡片样式 */
    .result-card {
        background-color: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.06);
        margin-bottom: 1.5rem;
        border-left: 6px solid #0066cc;
        text-align: center;
    }
    .metric-value {
        font-size: 3rem;
        font-weight: 800;
        color: #0066cc;
        line-height: 1.2;
    }
    .metric-unit {
        font-size: 1.25rem;
        color: #475569;
        font-weight: 500;
    }
    
    /* 标题样式优化 */
    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
    
    /* 面板样式 */
    .stExpander {
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    .stExpander > div > div:first-child {
        border-radius: 12px;
        background-color: white;
    }
    
    /* 输入框样式 */
    .stNumberInput, .stSelectbox {
        margin-bottom: 0.75rem;
    }
    
    /* 提示卡片 */
    .info-card {
        background-color: #f1f5f9;
        padding: 1.25rem;
        border-radius: 12px;
        border-left: 4px solid #0099ff;
        margin-top: 1rem;
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

# ================== 中文字体配置 ==================
font_path = "wqy-microhei.ttf"
try:
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
    plt.rcParams['axes.unicode_minus'] = False
except:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

# ================== 城市宏观数据预设 ==================
CITY_MACRO = {
    "济南市": {"income": 62506, "gdp": 135200, "population": 943.7, "tertiary": 62.8},
    "烟台市": {"income": 59126, "gdp": 144241, "population": 703.22, "tertiary": 51.0},
    "济宁市": {"income": 45055, "gdp": 66741, "population": 824.05, "tertiary": 51.03}
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

# ========== 【重点优化】SHAP瀑布图 更直观、配色分明、字体加大 ==========
def plot_shap_waterfall(input_dict):
    input_df = pd.DataFrame([input_dict])[FEATURE_COLS]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    
    plt.clf()
    # 加大画布，宽屏展示
    fig = plt.figure(figsize=(14, 7), dpi=150)
    
    # 自定义配色：红色=正向抬价，蓝色=负向降价
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=input_df.iloc[0].values,
            feature_names=FEATURE_COLS
        ),
        show=False,
        max_display=15,   # 多展示关键因子
        # 配色强化区分
    )

    # 全局字体放大
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.title("房价关键影响因素分解 | 红色=提升房价  蓝色=拉低房价", fontsize=12, pad=15, color="#2c3e50")
    plt.tight_layout(pad=2.0)
    return fig

# ================== 初始化 session_state 默认值 ==================
if 'init_done' not in st.session_state:
    # 房屋本体
    st.session_state.area = 100.0
    st.session_state.age = 5
    st.session_state.orientation = '南'
    st.session_state.decoration = '精装'
    st.session_state.elevator = '有'
    # 微观区位
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
    # 宏观
    default_macro = CITY_MACRO['济南市']
    st.session_state.income = default_macro['income']
    st.session_state.gdp = default_macro['gdp']
    st.session_state.population = default_macro['population']
    st.session_state.tertiary = default_macro['tertiary']
    st.session_state.init_done = True

# ================== 页面标题 ==================
st.markdown("<h1 style='text-align: center; color: #1e293b; margin-bottom: 2rem;'>🏠 房价预测与影响因素分析系统</h1>", unsafe_allow_html=True)

# ================== 宏观指标模块 ==================
st.markdown("<div class='section-title'>📊 城市宏观经济指标</div>", unsafe_allow_html=True)
with st.container():
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

# ================== 主界面：左侧输入，右侧结果 ==================
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
    
    st.markdown("<div class='section-title' style='margin-top: 1.5rem;'>🚏 周边配套设施</div>", unsafe_allow_html=True)
    
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
    
    # 预测按钮
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
            with st.spinner("模型正在计算预测..."):
                pred = predict_price(input_dict)
                fig = plot_shap_waterfall(input_dict)
            st.session_state['pred'] = pred
            st.session_state['fig'] = fig
        except Exception as e:
            st.error(f"预测失败: {str(e)}")
    
    # 显示预测结果
    if 'pred' in st.session_state:
        pred = st.session_state.pred
        
        # 预测价格展示
        st.markdown(f'''
        <div class="result-card">
            <div class="metric-value">{pred:.0f}</div>
            <div class="metric-unit">元/平方米</div>
        </div>
        ''', unsafe_allow_html=True)
        
        # 模型信息
        with st.container():
            st.caption(f"📊 训练集基准均价：{y_train_mean:.0f} 元/平米")
            st.caption(f"⚙️ 模型预测误差：{train_rmse:.0f} 元/平米 | 相对误差：{train_mape_percent:.1f}%")
        
        # SHAP分析结果
        st.markdown("<div class='section-title' style='margin-top: 1.5rem;'>🔍 价格影响因素分析</div>", unsafe_allow_html=True)
        st.pyplot(st.session_state.fig, use_container_width=True)
        
        # 下载功能
        buf = BytesIO()
        st.session_state.fig.savefig(buf, format="png", dpi=150, bbox_inches='tight')
        buf.seek(0)
        st.download_button(
            label="📥 下载因素分析图",
            data=buf,
            file_name="房价影响因素分析.png",
            mime="image/png",
            use_container_width=True
        )
    else:
        st.markdown("""
        <div class="info-card">
            <h4 style='margin-top:0; color:#0066cc;'>💡 使用指南</h4>
            <p style='margin-bottom:0; line-height:1.6;'>
            1. 填写上方的房屋基础信息<br>
            2. 完善周边配套设施参数<br>
            3. 点击「开始预测房价」按钮获取结果<br>
            4. 查看影响房价的关键因素分析
            </p>
        </div>
        """, unsafe_allow_html=True)

# ================== 页脚 ==================
st.markdown("---")
st.markdown("<p style='text-align: center; color: #64748b;'>© 2025 房价预测系统 | 基于XGBoost+SHAP | 数据来源：2021-2024 山东省济南/济宁/烟台</p>", unsafe_allow_html=True)
