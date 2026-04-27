import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu

# ================== 页面配置 ==================
st.set_page_config(
    page_title="房价预测与智能分析系统",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== 修复中文显示 ==================
font_path = "wqy-microhei.ttf"
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
plt.rcParams['axes.unicode_minus'] = False

# ================== 加载模型 ==================
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_xgboost_tuned.pkl")
    feature_cols = joblib.load("feature_columns.pkl")
    encoders = joblib.load("label_encoders.pkl")
    y_mean = np.load("y_train_mean.npy").item()
    return model, feature_cols, encoders, y_mean

try:
    model, FEATURE_COLS, encoders, y_train_mean = load_artifacts()
except FileNotFoundError as e:
    st.error(f"❌ 缺少必要文件：{e}")
    st.stop()

# ================== 辅助函数 ==================
def encode_categorical(value, encoder):
    try:
        return encoder.transform([value])[0]
    except:
        if '其他' in encoder.classes_:
            return encoder.transform(['其他'])[0]
        else:
            return encoder.transform([encoder.classes_[0]])[0]

def predict_price(input_dict):
    df = pd.DataFrame([input_dict])[FEATURE_COLS]
    return model.predict(df)[0]

def get_shap_values(input_dict):
    df = pd.DataFrame([input_dict])[FEATURE_COLS]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df)
    return shap_values[0], explainer.expected_value

# ================== 侧边栏导航 ==================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/home.png", width=80)
    st.title("🏠 智能房价助手")
    selected = option_menu(
        menu_title="导航",
        options=["房价预测", "特征分析", "关于"],
        icons=["house", "bar-chart", "info-circle"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "#1f77b4", "font-size": "20px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px"},
            "nav-link-selected": {"background-color": "#1f77b4"},
        }
    )

# ================== 城市与宏观数据选择 ==================
city = st.sidebar.selectbox("🌆 选择城市", ["济南市", "烟台市", "济宁市"])
# 城市对应的默认宏观数据（2023年）
city_macro = {
    "济南市": (62506, 135200, 943.7, 62.8),
    "烟台市": (59126, 144241, 703.22, 51.0),
    "济宁市": (45055, 66600, 824.05, 51.03)
}
default_income, default_gdp, default_pop, default_ter = city_macro[city]

with st.sidebar.expander("📊 宏观经济指标（可手动调整）", expanded=False):
    income = st.number_input("城镇居民人均可支配收入 (元)", value=default_income, step=1000)
    gdp = st.number_input("人均GDP (元)", value=default_gdp, step=5000)
    population = st.number_input("常住人口 (万人)", value=default_pop, step=10.0)
    tertiary = st.number_input("第三产业占比 (%)", value=default_ter, step=1.0)

# ================== 主页面 ==================
if selected == "房价预测":
    st.title("🏙️ 二手房价格预测与智能归因")
    st.markdown("> 输入房屋属性，AI将预测单价并解释关键影响因素")

    # 三列布局，增加视觉层次
    col_left, col_mid, col_right = st.columns([1, 1.2, 1.5], gap="large")

    with col_left:
        with st.container():
            st.markdown("### 🏷️ 房屋本体")
            area = st.slider("建筑面积 (㎡)", 30, 300, 100, 1, format="%d ㎡")
            age = st.slider("房龄 (年)", 0, 50, 5, 1, format="%d 年")
            orientation = st.selectbox("朝向", ["南", "北", "东", "西", "其他"])
            decoration = st.selectbox("装修程度", ["精装", "简装", "毛坯", "其他"])
            elevator = st.selectbox("有无电梯", ["有", "无"])

    with col_mid:
        st.markdown("### 🚇 微观区位")
        with st.expander("交通设施", expanded=True):
            dist_subway = st.number_input("🚇 距最近地铁站 (米)", 0, 20000, 1500, 100)
            count_subway = st.number_input("📌 10km内地铁站数", 0, 20, 2)
            dist_bus = st.number_input("🚌 距最近公交站 (米)", 0, 5000, 500, 100)
            count_bus = st.number_input("📌 10km内公交站数", 0, 100, 10, 5)
        with st.expander("教育·医疗·商业", expanded=False):
            dist_school = st.number_input("🏫 距最近学校 (米)", 0, 10000, 1000, 100)
            count_school = st.number_input("📌 10km内学校数", 0, 50, 5)
            dist_hospital = st.number_input("🏥 距最近综合医院 (米)", 0, 15000, 2000, 100)
            count_hospital = st.number_input("📌 10km内综合医院数", 0, 20, 2)
            dist_clinic = st.number_input("💊 距最近诊所 (米)", 0, 5000, 800, 100)
            count_clinic = st.number_input("📌 10km内诊所数", 0, 50, 4)
            dist_pharmacy = st.number_input("💊 距最近药店 (米)", 0, 2000, 300, 100)
            count_pharmacy = st.number_input("📌 10km内药店数", 0, 100, 8)
        with st.expander("商业·餐饮·公园", expanded=False):
            dist_mall = st.number_input("🛍️ 距最近大型商场 (米)", 0, 10000, 1500, 100)
            count_mall = st.number_input("📌 10km内大型商场数", 0, 20, 1)
            dist_small_business = st.number_input("🏪 距最近小型商业 (米)", 0, 3000, 200, 100)
            count_small_business = st.number_input("📌 10km内小型商业数", 0, 200, 20, 5)
            dist_catering = st.number_input("🍜 距最近餐饮 (米)", 0, 2000, 100, 50)
            count_catering = st.number_input("📌 10km内餐饮数", 0, 300, 30, 10)
            dist_park = st.number_input("🌳 距最近公园 (米)", 0, 10000, 1000, 100)
            count_park = st.number_input("📌 10km内公园数", 0, 20, 2)

    with col_right:
        st.markdown("### 📈 预测结果")
        predict_btn = st.button("🔮 开始预测", type="primary", use_container_width=True)
        
        if predict_btn:
            # 编码分类变量
            ori_enc = encode_categorical(orientation, encoders['朝向'])
            dec_enc = encode_categorical(decoration, encoders['装修'])
            ele_enc = encode_categorical(elevator, encoders['有无电梯'])
            
            input_dict = {
                '建筑面积': area, '房龄': age, '朝向': ori_enc, '装修': dec_enc, '有无电梯': ele_enc,
                'dist_地铁站': dist_subway, 'count_地铁站_within_10000m': count_subway,
                'dist_公交站': dist_bus, 'count_公交站_within_10000m': count_bus,
                'dist_学校': dist_school, 'count_学校_within_10000m': count_school,
                'dist_综合医院': dist_hospital, 'count_综合医院_within_10000m': count_hospital,
                'dist_诊所/社区医院': dist_clinic, 'count_诊所/社区医院_within_10000m': count_clinic,
                'dist_药店': dist_pharmacy, 'count_药店_within_10000m': count_pharmacy,
                'dist_大型商场': dist_mall, 'count_大型商场_within_10000m': count_mall,
                'dist_小型商业': dist_small_business, 'count_小型商业_within_10000m': count_small_business,
                'dist_餐饮': dist_catering, 'count_餐饮_within_10000m': count_catering,
                'dist_公园': dist_park, 'count_公园_within_10000m': count_park,
                '城镇居民人均可支配收入': income, '人均GDP': gdp, '常住人口': population, '第三产业占比': tertiary
            }
            
            with st.spinner("🏋️ 模型推理中..."):
                pred = predict_price(input_dict)
                shap_vals, base_val = get_shap_values(input_dict)
            
            # 展示预测值（仪表盘风格）
            col1, col2 = st.columns(2)
            with col1:
                st.metric("预测单价 (元/㎡)", f"{pred:.0f}", delta=f"较基准 {pred - y_train_mean:.0f}")
            with col2:
                st.metric("基准值 (训练集均价)", f"{y_train_mean:.0f}")
            
            # 进度条显示置信度（模拟）
            confidence = min(100, max(0, 100 - abs(pred - y_train_mean)/y_train_mean*50))
            st.progress(int(confidence), text=f"模型置信度 {confidence:.0f}%")
            
            # SHAP瀑布图
            st.subheader("🔍 影响因素解读")
            fig = plt.figure(figsize=(12, 5))
            shap.waterfall_plot(
                shap.Explanation(values=shap_vals, base_values=base_val, data=np.array(list(input_dict.values())), feature_names=FEATURE_COLS),
                show=False, max_display=12
            )
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
            # 额外显示最重要的前5个特征贡献（表格）
            st.subheader("📊 特征贡献分解")
            contrib_df = pd.DataFrame({
                "特征": FEATURE_COLS,
                "SHAP值": shap_vals
            }).sort_values("SHAP值", key=abs, ascending=False).head(5)
            contrib_df["影响方向"] = contrib_df["SHAP值"].apply(lambda x: "🔺 推高价格" if x > 0 else "🔻 拉低价格")
            contrib_df["SHAP值"] = contrib_df["SHAP值"].apply(lambda x: f"{x:.0f}")
            st.dataframe(contrib_df, use_container_width=True, hide_index=True)
        else:
            st.info("👈 请填写房屋特征，然后点击「开始预测」按钮。")
            st.image("https://img.icons8.com/fluency/240/homepage.png", width=200)

# ================== 特征分析页面 ==================
elif selected == "特征分析":
    st.title("📊 模型特征重要性分析")
    st.markdown("基于SHAP全局重要性，展示哪些因素对房价影响最大。")
    
    # 加载一个示例SHAP值（预先计算一个代表样本）
    # 这里简单展示预定义的特征重要性图（可从你的shap_summary.png读取）
    if st.button("📈 显示全局特征重要性图"):
        try:
            st.image("shap_summary.png", caption="SHAP 全局特征重要性", use_container_width=True)
        except:
            st.warning("未找到 shap_summary.png，请先运行SHAP分析生成该文件。")
    
    # 展示特征含义解释
    st.subheader("📖 特征含义说明")
    feature_desc = {
        "count_地铁站_within_10000m": "10公里内地铁站数量，反映公共交通便利性。",
        "建筑面积": "房屋建筑面积，面积越大总价越高，但单价可能边际递减。",
        "count_诊所/社区医院_within_10000m": "周边基层医疗配套，影响生活便利度。",
        "count_综合医院_within_10000m": "大型医疗资源，对房价有显著正向作用。",
        "count_餐饮_within_10000m": "餐饮丰富度，体现生活便利和人气。",
        "城镇居民人均可支配收入": "城市居民收入水平，决定购买力。",
        "人均GDP": "经济发展程度，间接影响房价水平。"
    }
    for feature, desc in feature_desc.items():
        st.markdown(f"- **{feature}**：{desc}")

# ================== 关于页面 ==================
else:
    st.title("📖 关于系统")
    st.markdown("""
    **房价预测与智能分析系统**  
    基于 XGBoost 机器学习模型，融合房屋本体、微观区位与宏观经济数据，实现二手房单价预测和可解释性分析。
    
    - **模型精度**：R² ≈ 0.80，RMSE ≈ 3000 元/㎡  
    - **解释框架**：SHAP (Shapley Additive Explanations)  
    - **数据来源**：济南、烟台、济宁三市2021–2024年二手房挂牌数据 + 高德POI + 统计年鉴  
    
    **开发者**：周奕彤 中国农业大学  
    **导师**：朱玲  
    **版本**：v2.0 (优化版)
    """)
    st.image("https://img.icons8.com/color/96/xgboost.png", width=80)
