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
    /* 卡片样式 */
    .css-1r6slb0, .stApp {
        background-color: #f8f9fa;
    }
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
    .help-icon {
        color: #888;
        font-size: 0.8rem;
        cursor: help;
    }
    hr {
        margin: 1rem 0;
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
    # 训练集误差标准差（用于模拟置信区间，实际可从训练过程获取）
    train_rmse = 2988.51  # 从之前实验结果填入
except FileNotFoundError as e:
    st.error(f"❌ 缺少必要的模型文件：{e}\n请确保 best_xgboost_tuned.pkl, feature_columns.pkl, label_encoders.pkl, y_train_mean.npy 在当前目录下。")
    st.stop()

# ================== 中文字体配置 ==================
font_path = "wqy-microhei.ttf"  # 请确保字体文件存在
try:
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
    plt.rcParams['axes.unicode_minus'] = False
except:
    st.warning("字体文件未找到，SHAP图中文字符可能显示异常。")

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
    """随机选择一个典型房源样本（可根据实际数据构造）"""
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

# ================== 页面标题与说明 ==================
st.title("🏠 房价预测与影响因素分析系统")
st.markdown("基于 **XGBoost 集成学习模型**与 **SHAP 可解释性框架**，为您提供二手房单价预测及关键驱动因素解读。")

# ================== 侧边栏快速设置 ==================
with st.sidebar:
    st.header("⚙️ 全局设置")
    city_choice = st.selectbox("选择城市（自动填充宏观指标）", list(CITY_MACRO.keys()), index=0)
    reset_macro = st.button("📊 重置宏观指标为当前城市默认值")
    st.markdown("---")
    st.subheader("🎲 快速测试")
    if st.button("📋 填充示例房源"):
        sample = get_sample_input()
        st.session_state['sample_loaded'] = sample
        st.success("示例已加载，请切换到主界面查看")
    st.markdown("---")
    st.caption("提示：鼠标悬停在输入框上可查看说明。")

# ================== 宏观指标初始化 ==================
if 'income' not in st.session_state or reset_macro:
    default = CITY_MACRO[city_choice]
    st.session_state.income = default["income"]
    st.session_state.gdp = default["gdp"]
    st.session_state.population = default["population"]
    st.session_state.tertiary = default["tertiary"]

# ================== 宏观指标（可编辑） ==================
with st.expander("📊 宏观经济指标（可手动调整）", expanded=False):
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        income = st.number_input("城镇居民人均可支配收入 (元)", min_value=30000, max_value=100000, value=st.session_state.income, step=1000, key="income_input", help="2023年济南市：62506元")
    with col_m2:
        gdp = st.number_input("人均GDP (元)", min_value=50000, max_value=200000, value=st.session_state.gdp, step=5000, key="gdp_input")
    with col_m3:
        population = st.number_input("常住人口 (万人)", min_value=500.0, max_value=1200.0, value=st.session_state.population, step=10.0, key="pop_input")
    with col_m4:
        tertiary = st.number_input("第三产业占比 (%)", min_value=40.0, max_value=80.0, value=st.session_state.tertiary, step=1.0, key="ter_input")

# ================== 主界面：两列布局（左侧输入，右侧结果） ==================
col_left, col_right = st.columns([1.2, 1.5], gap="large")

with col_left:
    # 房屋本体特征卡片
    with st.container():
        st.markdown("### 🏷️ 房屋本体属性")
        area = st.number_input("建筑面积 (㎡)", min_value=30.0, max_value=300.0, value=100.0, step=1.0, help="房屋的建筑面积，单位平方米")
        age = st.number_input("房龄 (年)", min_value=0, max_value=50, value=5, step=1, help="房屋竣工至今的年限")
        col_o, col_d, col_e = st.columns(3)
        with col_o:
            orientation = st.selectbox("朝向", ["南", "北", "东", "西", "其他"], help="主要房间朝向")
        with col_d:
            decoration = st.selectbox("装修程度", ["精装", "简装", "毛坯", "其他", "未知"])
        with col_e:
            elevator = st.selectbox("有无电梯", ["有", "无", "未知"])
    
    st.markdown("---")
    
    # 微观区位特征（使用折叠分组）
    with st.expander("🚉 交通设施", expanded=False):
        dist_subway = st.number_input("距最近地铁站 (米)", min_value=0, max_value=20000, value=1500, step=100)
        count_subway = st.number_input("10km内地铁站数", min_value=0, max_value=20, value=2, step=1)
        dist_bus = st.number_input("距最近公交站 (米)", min_value=0, max_value=5000, value=500, step=100)
        count_bus = st.number_input("10km内公交站数", min_value=0, max_value=100, value=10, step=5)
    
    with st.expander("📚 教育医疗", expanded=False):
        dist_school = st.number_input("距最近学校 (米)", min_value=0, max_value=10000, value=1000, step=100)
        count_school = st.number_input("10km内学校数", min_value=0, max_value=50, value=5, step=1)
        dist_hospital = st.number_input("距最近综合医院 (米)", min_value=0, max_value=15000, value=2000, step=100)
        count_hospital = st.number_input("10km内综合医院数", min_value=0, max_value=20, value=2, step=1)
        dist_clinic = st.number_input("距最近诊所/社区医院 (米)", min_value=0, max_value=5000, value=800, step=100)
        count_clinic = st.number_input("10km内诊所/社区医院数", min_value=0, max_value=50, value=4, step=1)
        dist_pharmacy = st.number_input("距最近药店 (米)", min_value=0, max_value=2000, value=300, step=100)
        count_pharmacy = st.number_input("10km内药店数", min_value=0, max_value=100, value=8, step=1)
    
    with st.expander("🛍️ 商业休闲", expanded=False):
        dist_mall = st.number_input("距最近大型商场 (米)", min_value=0, max_value=10000, value=1500, step=100)
        count_mall = st.number_input("10km内大型商场数", min_value=0, max_value=20, value=1, step=1)
        dist_small_business = st.number_input("距最近小型商业 (米)", min_value=0, max_value=3000, value=200, step=100)
        count_small_business = st.number_input("10km内小型商业数", min_value=0, max_value=200, value=20, step=5)
        dist_catering = st.number_input("距最近餐饮场所 (米)", min_value=0, max_value=2000, value=100, step=50)
        count_catering = st.number_input("10km内餐饮数", min_value=0, max_value=300, value=30, step=10)
        dist_park = st.number_input("距最近公园 (米)", min_value=0, max_value=10000, value=1000, step=100)
        count_park = st.number_input("10km内公园数", min_value=0, max_value=20, value=2, step=1)

with col_right:
    st.markdown("## 📈 预测结果")
    predict_btn = st.button("🔮 开始预测", type="primary", use_container_width=True)
    
    # 用于存储预测结果和图表
    if 'pred_price' not in st.session_state:
        st.session_state.pred_price = None
        st.session_state.fig = None
    
    if predict_btn:
        try:
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
            with st.spinner("模型推理中..."):
                pred = predict_price(input_dict)
                st.session_state.pred_price = pred
                # 生成SHAP图
                fig = plot_shap_waterfall(input_dict)
                st.session_state.fig = fig
                st.success(f"**预测单价：{pred:.0f} 元/平米**")
                # 简单置信区间（鲁棒性示意）
                lower = pred - 1.96 * train_rmse
                upper = pred + 1.96 * train_rmse
                st.caption(f"95% 置信区间：[{lower:.0f}, {upper:.0f}] 元/平米")
        except Exception as e:
            st.error(f"预测失败：{str(e)}")
    
    # 显示结果卡片
    if st.session_state.pred_price is not None:
        with st.container():
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown(f"### 💰 预测单价")
            st.markdown(f'<span class="metric-value">{st.session_state.pred_price:.0f}</span> <span style="font-size:1.2rem;">元/平米</span>', unsafe_allow_html=True)
            st.markdown(f"**🔹 基准值**：{y_train_mean:.0f} 元/平米（训练集均价）")
            st.markdown(f"**📊 预测误差范围**：±{1.96*train_rmse:.0f} 元/平米 (95%置信)")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # SHAP图展示及下载
        if st.session_state.fig is not None:
            st.subheader("🔍 影响因素贡献分解")
            st.pyplot(st.session_state.fig)
            # 下载按钮
            buf = BytesIO()
            st.session_state.fig.savefig(buf, format="png", dpi=150, bbox_inches='tight')
            buf.seek(0)
            st.download_button("📥 下载SHAP图", data=buf, file_name="shap_waterfall.png", mime="image/png")
    else:
        st.info("👈 请填写左侧房屋特征，然后点击「开始预测」按钮。")
        # 显示帮助提示
        st.markdown("""
        <div style="background-color: #eef2f7; padding: 1rem; border-radius: 15px; margin-top: 1rem;">
        <p>💡 <strong>使用小贴士</strong></p>
        <ul>
        <li>宏观指标已预置济南市2023年数据，可在上方展开修改。</li>
        <li>微观区位特征可按类别逐个填写，或使用侧边栏「填充示例」快速测试。</li>
        <li>SHAP瀑布图将展示各特征对预测值的正向/负向影响。</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# ================== 页脚 ==================
st.markdown("---")
st.caption("© 2025 房价预测系统 | 模型基于 2021-2024 年山东省济南、烟台、济宁二手房数据训练 | SHAP解释基于测试集近似")
