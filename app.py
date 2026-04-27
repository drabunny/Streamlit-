import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.preprocessing import LabelEncoder

# ================== 页面配置 ==================
st.set_page_config(
    page_title="房价预测与影响因素分析系统",
    page_icon="🏠",
    layout="wide"
)

font_path = "simhei.ttf"
fm.fontManager.addfont(font_path)
zh_font = fm.FontProperties(fname=font_path)

plt.rcParams['font.sans-serif'] = ['simhei', 'DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
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
except FileNotFoundError as e:
    st.error(f"❌ 缺少必要的模型文件：{e}\n请确保 best_xgboost_tuned.pkl, feature_columns.pkl, label_encoders.pkl, y_train_mean.npy 在当前目录下。")
    st.stop()

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
    """生成 SHAP 瀑布图，返回 figure 对象"""
    input_df = pd.DataFrame([input_dict])[FEATURE_COLS]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    
    plt.clf()  # 清除旧图形，防止重叠
    fig = plt.figure(figsize=(12, 6), dpi=100)
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=input_df.iloc[0].values,
            feature_names=FEATURE_COLS  # 此处返回的是中文列表，图表将显示中文
        ),
        show=False,
        max_display=15
    )
    plt.tight_layout()
    return fig

# ================== 页面标题 ==================
st.title("🏠 房价预测与影响因素分析系统")
st.markdown("基于 XGBoost 模型的二手房单价预测，并利用 SHAP 方法解释各特征的影响。")

# ================== 宏观经济指标（可选） ==================
with st.expander("📊 宏观经济指标（可选，默认使用济南市2023年数据）", expanded=False):
    col_macro1, col_macro2, col_macro3, col_macro4 = st.columns(4)
    with col_macro1:
        income = st.number_input("城镇居民人均可支配收入 (元)", min_value=30000, max_value=100000, value=62506, step=1000)
    with col_macro2:
        gdp = st.number_input("人均GDP (元)", min_value=50000, max_value=200000, value=135200, step=5000)
    with col_macro3:
        population = st.number_input("常住人口 (万人)", min_value=500.0, max_value=1200.0, value=943.7, step=10.0)
    with col_macro4:
        tertiary = st.number_input("第三产业占比 (%)", min_value=40.0, max_value=80.0, value=62.8, step=1.0)

# ================== 主界面：三列布局 ==================
col_left, col_mid, col_right = st.columns([1, 1.2, 1.5], gap="medium")

with col_left:
    st.subheader("🏷️ 房屋本体属性")
    area = st.number_input("建筑面积 (㎡)", min_value=30.0, max_value=300.0, value=100.0, step=1.0)
    age = st.number_input("房龄 (年)", min_value=0, max_value=50, value=5, step=1)
    orientation = st.selectbox("朝向", ["南", "北", "东", "西", "其他"])
    decoration = st.selectbox("装修程度", ["精装", "简装", "毛坯", "其他", "未知"])
    elevator = st.selectbox("有无电梯", ["有", "无", "未知"])

with col_mid:
    st.subheader("🚉 微观区位特征")
    st.caption("距离单位均为米（m）")
    dist_subway = st.number_input("🚇 距最近地铁站 (米)", min_value=0, max_value=20000, value=1500, step=100)
    count_subway = st.number_input("📌 10km内地铁站数", min_value=0, max_value=20, value=2, step=1)
    dist_bus = st.number_input("🚌 距最近公交站 (米)", min_value=0, max_value=5000, value=500, step=100)
    count_bus = st.number_input("📌 10km内公交站数", min_value=0, max_value=100, value=10, step=5)
    dist_school = st.number_input("🏫 距最近学校 (米)", min_value=0, max_value=10000, value=1000, step=100)
    count_school = st.number_input("📌 10km内学校数", min_value=0, max_value=50, value=5, step=1)
    dist_hospital = st.number_input("🏥 距最近综合医院 (米)", min_value=0, max_value=15000, value=2000, step=100)
    count_hospital = st.number_input("📌 10km内综合医院数", min_value=0, max_value=20, value=2, step=1)
    dist_clinic = st.number_input("💊 距最近诊所/社区医院 (米)", min_value=0, max_value=5000, value=800, step=100)
    count_clinic = st.number_input("📌 10km内诊所/社区医院数", min_value=0, max_value=50, value=4, step=1)
    dist_pharmacy = st.number_input("💊 距最近药店 (米)", min_value=0, max_value=2000, value=300, step=100)
    count_pharmacy = st.number_input("📌 10km内药店数", min_value=0, max_value=100, value=8, step=1)
    dist_mall = st.number_input("🛍️ 距最近大型商场 (米)", min_value=0, max_value=10000, value=1500, step=100)
    count_mall = st.number_input("📌 10km内大型商场数", min_value=0, max_value=20, value=1, step=1)
    dist_small_business = st.number_input("🏪 距最近小型商业 (米)", min_value=0, max_value=3000, value=200, step=100)
    count_small_business = st.number_input("📌 10km内小型商业数", min_value=0, max_value=200, value=20, step=5)
    dist_catering = st.number_input("🍜 距最近餐饮场所 (米)", min_value=0, max_value=2000, value=100, step=50)
    count_catering = st.number_input("📌 10km内餐饮数", min_value=0, max_value=300, value=30, step=10)
    dist_park = st.number_input("🌳 距最近公园 (米)", min_value=0, max_value=10000, value=1000, step=100)
    count_park = st.number_input("📌 10km内公园数", min_value=0, max_value=20, value=2, step=1)

with col_right:
    st.subheader("📈 预测结果")
    predict_btn = st.button("🔮 开始预测", type="primary", use_container_width=True)
    
    if predict_btn:
        # 编码分类变量
        ori_encoded = encode_categorical(orientation, encoders['朝向'])
        dec_encoded = encode_categorical(decoration, encoders['装修'])
        ele_encoded = encode_categorical(elevator, encoders['有无电梯'])
        
        # 构造输入字典
        input_dict = {
            '建筑面积': area,
            '房龄': age,
            '朝向': ori_encoded,
            '装修': dec_encoded,
            '有无电梯': ele_encoded,
            'dist_地铁站': dist_subway,
            'count_地铁站_within_10000m': count_subway,
            'dist_公交站': dist_bus,
            'count_公交站_within_10000m': count_bus,
            'dist_学校': dist_school,
            'count_学校_within_10000m': count_school,
            'dist_综合医院': dist_hospital,
            'count_综合医院_within_10000m': count_hospital,
            'dist_诊所/社区医院': dist_clinic,
            'count_诊所/社区医院_within_10000m': count_clinic,
            'dist_药店': dist_pharmacy,
            'count_药店_within_10000m': count_pharmacy,
            'dist_大型商场': dist_mall,
            'count_大型商场_within_10000m': count_mall,
            'dist_小型商业': dist_small_business,
            'count_小型商业_within_10000m': count_small_business,
            'dist_餐饮': dist_catering,
            'count_餐饮_within_10000m': count_catering,
            'dist_公园': dist_park,
            'count_公园_within_10000m': count_park,
            '城镇居民人均可支配收入': income,
            '人均GDP': gdp,
            '常住人口': population,
            '第三产业占比': tertiary
        }
        
        # 预测
        with st.spinner("模型推理中..."):
            pred = predict_price(input_dict)
        st.success(f"🏷️ **预测单价：{pred:.0f} 元/平米**")
        st.caption(f"基准值（训练集平均单价）：{y_train_mean:.0f} 元/平米")
        
        # SHAP瀑布图
        st.subheader("🔍 影响因素解读")
        with st.spinner("生成SHAP解释图..."):
            try:
                fig = plot_shap_waterfall(input_dict)
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.error(f"生成SHAP图时出错：{str(e)}")
                st.info("提示：预测功能正常，图表生成可能因环境缺少依赖而失败。")
    else:
        st.info("👈 请填写左侧特征，然后点击「开始预测」按钮。")
    
    # 说明注释放置在结果列底部
    st.markdown("---")
    st.caption("注：预测基于 XGBoost 模型（2021-2024年山东省济南/烟台/济宁数据），SHAP瀑布图显示各特征对预测值的影响贡献。")
