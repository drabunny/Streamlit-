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
    .stApp { background-color: #f5f7fa; }
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
    .metric-value { font-size: 3rem; font-weight: 800; color: #1677ff; line-height: 1.1; }
    .metric-unit { font-size: 1.25rem; color: #4e5969; font-weight: 500; }
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
    .advice-card {
        background-color: #f0f9ff;
        padding: 1.25rem;
        border-radius: 12px;
        border-left: 4px solid #52c41a;
        margin-top: 1rem;
    }
    hr { border-color: #f0f0f0; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ================== 加载模型与处理对象 ==================
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model_final_deploy.pkl")
    feature_cols = joblib.load("feature_columns_final_deploy.pkl")
    encoders = joblib.load("label_encoders_final_deploy.pkl")
    y_mean = np.load("y_train_log_mean.npy").item()
    return model, feature_cols, encoders, y_mean

try:
    model, FEATURE_COLS, encoders, y_train_log_mean = load_artifacts()
except FileNotFoundError as e:
    st.error(f"❌ 缺少必要的模型文件：{e}")
    st.stop()

# ================== 中文字体 ==================
font_path = 'wqy-microhei.ttf'
try:
    fm.fontManager.addfont(font_path)
except:
    pass
plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name() if font_path else plt.rcParams['font.sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ================== 自动获取训练集城市列表 ==================
KNOWN_CITIES = None

# 尝试从 encoders 中获取城市编码器的类别
if '城市' in encoders:
    KNOWN_CITIES = encoders['城市'].classes_.tolist()
    st.info(f"🔍 从编码器中加载训练集城市列表：{KNOWN_CITIES}")
else:
    # 如果 encoders 中没有 '城市'，则尝试从模型内部获取（若模型是用 enable_categorical=True 训练的）
    try:
        # 获取模型的特征类型，如果是 'c' 表示类别，并试图获取 categories_
        # 注意：XGBoost 1.6+ 的 Booster 对象有 feature_types 和 feature_names
        booster = model.get_booster()
        feature_types = booster.feature_types
        if feature_types:
            # 找到 '城市' 特征的索引
            if '城市' in FEATURE_COLS:
                idx = FEATURE_COLS.index('城市')
                # 如果该特征类型是 'c'，尝试从 booster 获取 categories（可能不支持）
                # 这里留作扩展，实际常用方法是在训练时保存编码器
                pass
    except:
        pass

# 如果仍然没有获取到，则使用您手动指定的列表（请根据实际情况修改）
if KNOWN_CITIES is None:
    # 这里替换为您的训练集实际城市名，例如 ['青岛市', '潍坊市', ...]
    KNOWN_CITIES = ['烟台市', '济宁市']   # 请务必修改为正确的城市名
    st.warning(f"⚠️ 未能从模型文件中读取城市列表，使用默认列表：{KNOWN_CITIES}。请确保这些城市确实在训练集中，否则预测仍可能出错。")

# ================== 宏观数据字典 ==================
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

def get_default_macro(city, year):
    return MACRO_DATA.get((city, year), MACRO_DATA[("济南市", 2023)])

# ================== 辅助函数 ==================
def encode_categorical(value, encoder):
    try:
        return encoder.transform([value])[0]
    except ValueError:
        if '其他' in encoder.classes_:
            return encoder.transform(['其他'])[0]
        else:
            return encoder.transform([encoder.classes_[0]])[0]

def compute_derived_features(basic_dict, city, year):
    # 城市映射：如果不在训练集列表中，则替换为第一个已知城市
    if city not in KNOWN_CITIES:
        st.warning(f"⚠️ 您选择的城市“{city}”不在模型训练集中，系统自动使用“{KNOWN_CITIES[0]}”替代进行预测，结果仅供参考。")
        city = KNOWN_CITIES[0]
    d = basic_dict.copy()
    d['地铁便利性'] = 1.0 / (d['dist_地铁站'] + 1) * np.log1p(d['count_地铁站_within_10000m'])
    d['医疗资源'] = d['count_综合医院_within_10000m'] + d['count_诊所/社区医院_within_10000m']
    d['商业繁华度'] = d['count_大型商场_within_10000m'] + d['count_小型商业_within_10000m']
    d['人均GDP_log'] = np.log1p(d['人均GDP'])
    d['城市'] = city
    d['年份'] = year
    return d

def predict_price(full_input_dict):
    input_df = pd.DataFrame([full_input_dict])[FEATURE_COLS]
    if '城市' in input_df.columns:
        input_df['城市'] = input_df['城市'].astype('category')
    pred_log = model.predict(input_df)[0]
    return np.expm1(pred_log)

# ================== SHAP瀑布图 ==================
def plot_shap_waterfall_clean(full_input_dict):
    input_df = pd.DataFrame([full_input_dict])[FEATURE_COLS]
    if '城市' in input_df.columns:
        input_df['城市'] = input_df['城市'].astype('category')
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    expected_value = explainer.expected_value
    pred_price = predict_price(full_input_dict)

    fig = plt.figure(figsize=(12, 8), facecolor='white')
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=expected_value,
            data=input_df.iloc[0].values,
            feature_names=FEATURE_COLS
        ),
        show=False,
        max_display=15
    )
    ax = plt.gca()
    for text in ax.texts:
        txt = text.get_text()
        if any(key in txt.lower() for key in ['model output', 'base value', 'f(x)=', 'e[f(x)]=', 'output value']):
            text.set_visible(False)
    ax.set_title(f'房价影响因素分解图（对数尺度）\n模型预测对数：{model.predict(input_df)[0]:.4f} → 预测单价：{pred_price:.0f} 元/平米', fontsize=14, pad=20)
    return fig

# ================== 购房建议（贡献值转换为价格单位） ==================
def generate_advice_from_shap(pred_price, top_positive, top_negative, input_dict, city):
    scale = pred_price + 1   # 将对数贡献转换为近似价格贡献 (dp = (1+p)*dlog)
    advice_parts = []

    # 总体评价
    if pred_price > 20000:
        price_level = "较高"
    elif pred_price > 12000:
        price_level = "中等偏上"
    elif pred_price > 8000:
        price_level = "中等"
    else:
        price_level = "较低"
    advice_parts.append(f"📊 总体评价\n该房源预测单价为 {pred_price:.0f}元/平米，属于{price_level}水平。")

    # 正向因素
    if top_positive:
        pos_lines = []
        for name, val in top_positive[:5]:
            contrib_price = int(round(val * scale))   # 转换为价格贡献
            # 特征友好描述
            if name == '建筑面积':
                pos_lines.append(f"- **建筑面积**：{input_dict.get('建筑面积',0):.0f} ㎡，贡献约 **+{contrib_price}** 元/平米")
            elif name == '地铁便利性':
                pos_lines.append(f"- **地铁便利性**（综合距离与数量）：贡献约 **+{contrib_price}** 元/平米")
            elif name == '医疗资源':
                pos_lines.append(f"- **医疗资源**（医院+诊所密度）：贡献约 **+{contrib_price}** 元/平米")
            elif name == '商业繁华度':
                pos_lines.append(f"- **商业繁华度**（商场+商业密度）：贡献约 **+{contrib_price}** 元/平米")
            elif 'count_地铁站' in name:
                pos_lines.append(f"- **地铁站数量**：10km内 {input_dict.get('count_地铁站_within_10000m',0)} 个，贡献约 **+{contrib_price}** 元/平米")
            elif 'count_公交站' in name:
                pos_lines.append(f"- **公交站密度**：10km内 {input_dict.get('count_公交站_within_10000m',0)} 个，贡献约 **+{contrib_price}** 元/平米")
            elif 'count_学校' in name:
                pos_lines.append(f"- **学校数量**：10km内 {input_dict.get('count_学校_within_10000m',0)} 所，贡献约 **+{contrib_price}** 元/平米")
            elif 'count_综合医院' in name:
                pos_lines.append(f"- **综合医院数量**：10km内 {input_dict.get('count_综合医院_within_10000m',0)} 家，贡献约 **+{contrib_price}** 元/平米")
            elif 'count_诊所/社区医院' in name:
                pos_lines.append(f"- **诊所/社区医院数量**：10km内 {input_dict.get('count_诊所/社区医院_within_10000m',0)} 家，贡献约 **+{contrib_price}** 元/平米")
            elif 'count_餐饮' in name:
                pos_lines.append(f"- **餐饮丰富度**：10km内 {input_dict.get('count_餐饮_within_10000m',0)} 家，贡献约 **+{contrib_price}** 元/平米")
            elif '城镇居民人均可支配收入' in name:
                pos_lines.append(f"- **居民收入**：{input_dict.get('城镇居民人均可支配收入',0):.0f} 元/年，贡献约 **+{contrib_price}** 元/平米")
            elif '人均GDP' in name:
                pos_lines.append(f"- **人均GDP**：{input_dict.get('人均GDP',0)} 元，贡献约 **+{contrib_price}** 元/平米")
            elif '常住人口' in name:
                pos_lines.append(f"- **人口规模**：{input_dict.get('常住人口',0)} 万人，贡献约 **+{contrib_price}** 元/平米")
            elif '第三产业占比' in name:
                pos_lines.append(f"- **服务业占比**：{input_dict.get('第三产业占比',0)}%，贡献约 **+{contrib_price}** 元/平米")
            else:
                pos_lines.append(f"- **{name}**：贡献约 **+{contrib_price}** 元/平米")
        advice_parts.append("### 📈 主要溢价因素\n" + "\n".join(pos_lines))

    # 负向因素
    if top_negative:
        neg_lines = []
        for name, val in top_negative[:5]:
            contrib_price = int(round(-val * scale))   # 取正数显示
            if name == '建筑面积':
                neg_lines.append(f"- **建筑面积**：{input_dict.get('建筑面积',0):.0f} ㎡，贡献约 **-{contrib_price}** 元/平米")
            elif name == '地铁便利性':
                neg_lines.append(f"- **地铁便利性**较低，贡献约 **-{contrib_price}** 元/平米")
            elif name == '医疗资源':
                neg_lines.append(f"- **医疗资源**不足，贡献约 **-{contrib_price}** 元/平米")
            elif name == '商业繁华度':
                neg_lines.append(f"- **商业繁华度**较低，贡献约 **-{contrib_price}** 元/平米")
            elif 'dist_地铁站' in name:
                dist = input_dict.get('dist_地铁站', 0)
                neg_lines.append(f"- **距地铁站** {dist} 米，较远，贡献约 **-{contrib_price}** 元/平米")
            elif 'dist_综合医院' in name:
                dist = input_dict.get('dist_综合医院', 0)
                neg_lines.append(f"- **距综合医院** {dist} 米，贡献约 **-{contrib_price}** 元/平米")
            elif 'dist_学校' in name:
                dist = input_dict.get('dist_学校', 0)
                neg_lines.append(f"- **距学校** {dist} 米，贡献约 **-{contrib_price}** 元/平米")
            elif '朝向' in name:
                ori_val = input_dict.get('朝向', '未知')
                ori_map = {0:"南",1:"北",2:"东",3:"西",4:"其他"}
                neg_lines.append(f"- **朝向** {ori_map.get(ori_val,'未知')}，贡献约 **-{contrib_price}** 元/平米")
            elif '装修' in name:
                dec_val = input_dict.get('装修', '未知')
                dec_map = {0:"精装",1:"简装",2:"毛坯",3:"其他",4:"未知"}
                neg_lines.append(f"- **装修** {dec_map.get(dec_val,'未知')}，贡献约 **-{contrib_price}** 元/平米")
            else:
                neg_lines.append(f"- **{name}**：贡献约 **-{contrib_price}** 元/平米")
        advice_parts.append("### 📉 主要折价因素\n" + "\n".join(neg_lines))

    # 城市洞察
    city_insight = {
        "济南市": "济南作为省会，长期发展潜力较好，地铁沿线或优质学区房源保值能力更强。",
        "烟台市": "烟台为沿海宜居城市，建议关注海景资源、旅游配套及开发区规划。",
        "济宁市": "济宁本地自住需求为主，房价相对平稳，可重点考察学校、医院周边房源。"
    }
    # 如果city被映射了，但原始city可能不在字典中，我们用映射后的city来取洞察
    advice_parts.append(f"### 🏙️ 城市洞察\n{city_insight.get(city, '根据当地市场情况综合判断。')}")

    # 综合建议
    if pred_price > 20000:
        purchase = "当前价格处于较高水平，建议仔细对比同地段类似房源，重点关注房屋质量及稀缺资源。"
    elif pred_price < 8000:
        purchase = "价格明显低于区域平均水平，性价比突出，但需谨慎排查房屋产权、质量隐患或周边不利设施。"
    else:
        purchase = "价格属于合理区间，可根据自身通勤需求、学区偏好及生活便利性做出决策。"
    advice_parts.append(f"### 💎 购买建议\n{purchase}")

    return "\n\n".join(advice_parts)

# ================== 初始化 session_state ==================
if 'init_done' not in st.session_state:
    st.session_state.area = 100.0
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
    st.session_state.city = "济南市"
    st.session_state.year = 2023
    default_macro = get_default_macro(st.session_state.city, st.session_state.year)
    st.session_state.income = default_macro["income"]
    st.session_state.gdp = default_macro["gdp"]
    st.session_state.population = default_macro["population"]
    st.session_state.tertiary = default_macro["tertiary"]
    st.session_state.init_done = True

# ================== 页面主标题 ==================
st.markdown("<h1 style='text-align: center; color: #1677ff; margin-bottom: 2rem;'>🏠 房价预测与影响因素分析系统</h1>", unsafe_allow_html=True)

# ================== 宏观经济指标 ==================
st.markdown("<div class='section-title'>📊 城市宏观经济指标</div>", unsafe_allow_html=True)
col_city, col_year, col_reset = st.columns([1, 1, 0.5])
with col_city:
    selected_city = st.selectbox("选择城市", ["济南市", "烟台市", "济宁市"], key="city_select")
with col_year:
    selected_year = st.selectbox("选择年份", [2021, 2022, 2023, 2024], key="year_select")
with col_reset:
    if st.button("🔄 重置", key="reset_macro"):
        default = get_default_macro(selected_city, selected_year)
        st.session_state.income = default["income"]
        st.session_state.gdp = default["gdp"]
        st.session_state.population = default["population"]
        st.session_state.tertiary = default["tertiary"]
        st.rerun()

if selected_city != st.session_state.city or selected_year != st.session_state.year:
    st.session_state.city = selected_city
    st.session_state.year = selected_year
    default = get_default_macro(selected_city, selected_year)
    st.session_state.income = default["income"]
    st.session_state.gdp = default["gdp"]
    st.session_state.population = default["population"]
    st.session_state.tertiary = default["tertiary"]

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.number_input("人均可支配收入 (元)", key="income", min_value=30000, max_value=100000, step=1000)
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
    col_a, col_b = st.columns(2)   # 只保留两列
    with col_a:
        st.number_input("建筑面积 (㎡)", key="area", min_value=30.0, max_value=300.0, step=1.0)
    with col_b:
        st.selectbox("房屋朝向", ["南", "北", "东", "西", "其他"], key="orientation")
    col_c, col_d = st.columns(2)
    with col_c:
        st.selectbox("装修程度", ["精装", "简装", "毛坯", "其他", "未知"], key="decoration")
    with col_d:
        st.selectbox("电梯配置", ["有", "无", "未知"], key="elevator")

    st.markdown("<div class='section-title'>🚏 周边配套设施</div>", unsafe_allow_html=True)

    with st.expander("🚇 交通设施", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.number_input("距最近地铁站 (米)", key="dist_subway", min_value=0, max_value=20000, step=100)
        with c2:
            st.number_input("10km内地铁站数量", key="count_subway", min_value=0, max_value=20000, step=1)
        c1, c2 = st.columns(2)
        with c1:
            st.number_input("距最近公交站 (米)", key="dist_bus", min_value=0, max_value=20000, step=100)
        with c2:
            st.number_input("10km内公交站数量", key="count_bus", min_value=0, max_value=20000, step=5)

    with st.expander("🏥 教育医疗"):
        c1, c2 = st.columns(2)
        with c1:
            st.number_input("距最近学校 (米)", key="dist_school", min_value=0, max_value=20000, step=100)
        with c2:
            st.number_input("10km内学校数量", key="count_school", min_value=0, max_value=20000, step=1)
        c1, c2 = st.columns(2)
        with c1:
            st.number_input("距最近综合医院 (米)", key="dist_hospital", min_value=0, max_value=20000, step=100)
        with c2:
            st.number_input("10km内综合医院数量", key="count_hospital", min_value=0, max_value=20000, step=1)
        c1, c2 = st.columns(2)
        with c1:
            st.number_input("距最近诊所 (米)", key="dist_clinic", min_value=0, max_value=20000, step=100)
        with c2:
            st.number_input("10km内诊所数量", key="count_clinic", min_value=0, max_value=20000, step=1)
        c1, c2 = st.columns(2)
        with c1:
            st.number_input("距最近药店 (米)", key="dist_pharmacy", min_value=0, max_value=20000, step=100)
        with c2:
            st.number_input("10km内药店数量", key="count_pharmacy", min_value=0, max_value=20000, step=1)

    with st.expander("🛍️ 商业休闲"):
        c1, c2 = st.columns(2)
        with c1:
            st.number_input("距最近大型商场 (米)", key="dist_mall", min_value=0, max_value=20000, step=100)
        with c2:
            st.number_input("10km内大型商场数量", key="count_mall", min_value=0, max_value=20000, step=1)
        c1, c2 = st.columns(2)
        with c1:
            st.number_input("距最近小型商业 (米)", key="dist_small_business", min_value=0, max_value=20000, step=100)
        with c2:
            st.number_input("10km内小型商业数量", key="count_small_business", min_value=0, max_value=20000, step=5)
        c1, c2 = st.columns(2)
        with c1:
            st.number_input("距最近餐饮场所 (米)", key="dist_catering", min_value=0, max_value=20000, step=50)
        with c2:
            st.number_input("10km内餐饮数量", key="count_catering", min_value=0, max_value=20000, step=10)
        c1, c2 = st.columns(2)
        with c1:
            st.number_input("距最近公园 (米)", key="dist_park", min_value=0, max_value=20000, step=100)
        with c2:
            st.number_input("10km内公园数量", key="count_park", min_value=0, max_value=20000, step=1)

with col_right:
    st.markdown("<div class='section-title'>📈 房价预测结果</div>", unsafe_allow_html=True)
    predict_btn = st.button("🔮 开始预测房价", type="primary", use_container_width=True)

    if predict_btn:
        try:
            # 构建基础特征（不含房龄）
            basic_dict = {
                '建筑面积': st.session_state.area,
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
            full_dict = compute_derived_features(basic_dict, st.session_state.city, st.session_state.year)

            with st.spinner("模型计算中，请稍候..."):
                pred_price = predict_price(full_dict)
                fig = plot_shap_waterfall_clean(full_dict)

                # SHAP 贡献（对数尺度）
                explainer = shap.TreeExplainer(model)
                input_df = pd.DataFrame([full_dict])[FEATURE_COLS]
                if '城市' in input_df.columns:
                    input_df['城市'] = input_df['城市'].astype('category')
                shap_vals = explainer.shap_values(input_df)
                feature_contrib = list(zip(FEATURE_COLS, shap_vals[0]))
                feature_contrib.sort(key=lambda x: x[1], reverse=True)
                top_positive = [(name, val) for name, val in feature_contrib if val > 0][:5]
                top_negative = [(name, val) for name, val in feature_contrib if val < 0][:5]

                advice = generate_advice_from_shap(pred_price, top_positive, top_negative, full_dict, st.session_state.city)

            st.session_state['pred'] = pred_price
            st.session_state['fig'] = fig
            st.session_state['advice'] = advice

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

        st.markdown("<div class='section-title'>💡购房建议</div>", unsafe_allow_html=True)
        st.markdown(f'<div class="advice-card">{st.session_state.advice}</div>', unsafe_allow_html=True)

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
            2. 点击「重置」按钮可将宏观数据恢复为默认值<br>
            3. 填写房屋基础属性与周边配套参数<br>
            4. 点击【开始预测房价】按钮，获取预测单价及因素分析
            </p>
        </div>
        """, unsafe_allow_html=True)

# ================== 页脚 ==================
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #86909c;'>房价预测系统 | XGBoost+SHAP | 数据范围：山东济南/烟台/济宁 2021-2024 | 购房建议由系统自动生成，仅供参考</p>",
    unsafe_allow_html=True
)
