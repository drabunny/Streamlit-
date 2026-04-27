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

# ================== 自定义CSS样式 ==================
st.markdown("""
<style>
    .stButton > button {
        background: linear-gradient(90deg, #0066cc, #0099ff);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #0055aa, #0088ee);
        box-shadow: 0 4px 12px rgba(0,102,204,0.3);
    }
    .result-card {
        background-color: white;
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 8px 20px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
        border-left: 5px solid #0066cc;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #0066cc;
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
    train_rmse = 2988.51  # 从实验中得到
    train_mape_percent = (train_rmse / y_train_mean) * 100  # 相对误差百分比
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
    pass

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

def plot_shap_waterfall(input_dict):
    input_df = pd.DataFrame([input_dict])[FEATURE_COLS]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    plt.clf()
    fig = plt.figure(figsize=(12, 5), dpi=100)
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=input_df.iloc[0].values,
            feature_names=FEATURE_COLS
        ),
        show=False,
        max_display=12
    )
    plt.tight_layout()
    return fig

def get_sample_input():
    """一个示例房源数据"""
    return {
        '建筑面积': 95.0,
        '房龄': 8,
        '朝向': '南',
        '装修': '精装',
        '有无电梯': '有',
        'dist_地铁站': 800,
        'count_地铁站_within_10000m': 3,
        'dist_公交站': 200,
        'count_公交站_within_10000m': 15,
        'dist_学校': 500,
        'count_学校_within_10000m': 6,
        'dist_综合医院': 1200,
        'count_综合医院_within_10000m': 2,
        'dist_诊所/社区医院': 400,
        'count_诊所/社区医院_within_10000m': 5,
        'dist_药店': 150,
        'count_药店_within_10000m': 10,
        'dist_大型商场': 2000,
        'count_大型商场_within_10000m': 1,
        'dist_小型商业': 100,
        'count_小型商业_within_10000m': 30,
        'dist_餐饮': 50,
        'count_餐饮_within_10000m': 50,
        'dist_公园': 600,
        'count_公园_within_10000m': 2
    }

# ================== 初始化 session_state 中的输入值 ==================
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

# ================== 侧边栏快速设置 ==================
with st.sidebar:
    st.header("⚙️ 快速测试")
    if st.button("📋 填充示例房源"):
        sample = get_sample_input()
        for key, value in sample.items():
            if key in st.session_state:
                st.session_state[key] = value
        st.success("示例已加载！请点击「开始预测」按钮")
        st.rerun()
    
    st.markdown("---")
    selected_city = st.selectbox("城市（自动更新宏观指标）", list(CITY_MACRO.keys()))
    if st.button("📊 应用该城市宏观指标"):
        macro = CITY_MACRO[selected_city]
        st.session_state.income = macro['income']
        st.session_state.gdp = macro['gdp']
        st.session_state.population = macro['population']
        st.session_state.tertiary = macro['tertiary']
        st.success(f"已应用{selected_city}的宏观数据")
        st.rerun()

# ================== 宏观指标（可编辑） ==================
with st.expander("📊 宏观经济指标（可手动调整）", expanded=False):
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.number_input("城镇居民人均可支配收入 (元)", key="income", min_value=30000, max_value=100000, step=1000)
    with col_m2:
        st.number_input("人均GDP (元)", key="gdp", min_value=50000, max_value=200000, step=5000)
    with col_m3:
        st.number_input("常住人口 (万人)", key="population", min_value=500.0, max_value=1200.0, step=10.0)
    with col_m4:
        st.number_input("第三产业占比 (%)", key="tertiary", min_value=40.0, max_value=80.0, step=1.0)

# ================== 主界面：左侧输入，右侧结果 ==================
col_left, col_right = st.columns([1.2, 1.5], gap="large")

with col_left:
    st.markdown("### 🏷️ 房屋本体属性")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.number_input("建筑面积 (㎡)", key="area", min_value=30.0, max_value=300.0, step=1.0)
    with col2:
        st.number_input("房龄 (年)", key="age", min_value=0, max_value=50, step=1)
    with col3:
        st.selectbox("朝向", ["南", "北", "东", "西", "其他"], key="orientation")
    col4, col5 = st.columns(2)
    with col4:
        st.selectbox("装修程度", ["精装", "简装", "毛坯", "其他", "未知"], key="decoration")
    with col5:
        st.selectbox("有无电梯", ["有", "无", "未知"], key="elevator")
    
    st.markdown("---")
    with st.expander("🚉 交通设施", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.number_input("距最近地铁站 (米)", key="dist_subway", min_value=0, max_value=20000, step=100)
        with c2:
            st.number_input("10km内地铁站数", key="count_subway", min_value=0, max_value=20, step=1)
        c1, c2 = st.columns(2)
        with c1:
            st.number_input("距最近公交站 (米)", key="dist_bus", min_value=0, max_value=5000, step=100)
        with c2:
            st.number_input("10km内公交站数", key="count_bus", min_value=0, max_value=100, step=5)
    
    with st.expander("📚 教育医疗", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.number_input("距最近学校 (米)", key="dist_school", min_value=0, max_value=10000, step=100)
        with c2:
            st.number_input("10km内学校数", key="count_school", min_value=0, max_value=50, step=1)
        c1, c2 = st.columns(2)
        with c1:
            st.number_input("距最近综合医院 (米)", key="dist_hospital", min_value=0, max_value=15000, step=100)
        with c2:
            st.number_input("10km内综合医院数", key="count_hospital", min_value=0, max_value=20, step=1)
        c1, c2 = st.columns(2)
        with c1:
            st.number_input("距最近诊所/社区医院 (米)", key="dist_clinic", min_value=0, max_value=5000, step=100)
        with c2:
            st.number_input("10km内诊所/社区医院数", key="count_clinic", min_value=0, max_value=50, step=1)
        c1, c2 = st.columns(2)
        with c1:
            st.number_input("距最近药店 (米)", key="dist_pharmacy", min_value=0, max_value=2000, step=100)
        with c2:
            st.number_input("10km内药店数", key="count_pharmacy", min_value=0, max_value=100, step=1)
    
    with st.expander("🛍️ 商业休闲", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.number_input("距最近大型商场 (米)", key="dist_mall", min_value=0, max_value=10000, step=100)
        with c2:
            st.number_input("10km内大型商场数", key="count_mall", min_value=0, max_value=20, step=1)
        c1, c2 = st.columns(2)
        with c1:
            st.number_input("距最近小型商业 (米)", key="dist_small_business", min_value=0, max_value=3000, step=100)
        with c2:
            st.number_input("10km内小型商业数", key="count_small_business", min_value=0, max_value=200, step=5)
        c1, c2 = st.columns(2)
        with c1:
            st.number_input("距最近餐饮场所 (米)", key="dist_catering", min_value=0, max_value=2000, step=50)
        with c2:
            st.number_input("10km内餐饮数", key="count_catering", min_value=0, max_value=300, step=10)
        c1, c2 = st.columns(2)
        with c1:
            st.number_input("距最近公园 (米)", key="dist_park", min_value=0, max_value=10000, step=100)
        with c2:
            st.number_input("10km内公园数", key="count_park", min_value=0, max_value=20, step=1)

with col_right:
    st.markdown("## 📈 预测结果")
    if st.button("🔮 开始预测", type="primary", use_container_width=True):
        # 收集所有输入
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
            with st.spinner("模型推理中..."):
                pred = predict_price(input_dict)
                fig = plot_shap_waterfall(input_dict)
            st.session_state['pred'] = pred
            st.session_state['fig'] = fig
        except Exception as e:
            st.error(f"预测失败: {e}")
    
    if 'pred' in st.session_state:
        pred = st.session_state.pred
        st.markdown(f'<div class="result-card"><span class="metric-value">{pred:.0f}</span> <span style="font-size:1.2rem;">元/平米</span></div>', unsafe_allow_html=True)
        st.caption(f"训练集均价基准：{y_train_mean:.0f} 元/平米")
        st.caption(f"模型预测误差 (RMSE)：{train_rmse:.0f} 元/平米，相对误差约 {train_mape_percent:.1f}%")
        st.subheader("🔍 影响因素贡献分解")
        st.pyplot(st.session_state.fig)
        buf = BytesIO()
        st.session_state.fig.savefig(buf, format="png", dpi=150, bbox_inches='tight')
        buf.seek(0)
        st.download_button("📥 下载SHAP图", data=buf, file_name="shap_waterfall.png", mime="image/png")
    else:
        st.info("👈 填写左侧特征后点击「开始预测」")
        st.markdown("""
        <div style="background-color: #eef2f7; padding: 1rem; border-radius: 15px;">
        <strong>💡 使用提示</strong><br>
        - 侧边栏可快速填充示例房源或切换城市宏观指标<br>
        - 宏观指标可展开手动调整<br>
        - 预测结果附带模型整体误差参考
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.caption("© 2025 房价预测系统 | 基于XGBoost+SHAP | 数据时间:2021-2024 山东省济南/济宁/烟台")
