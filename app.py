import os
import joblib
import json
import numpy as np
import pandas as pd
import streamlit as st

def inspect_city_categories():
    """
    检查模型和编码器中关于城市特征的信息，返回一个字典。
    """
    result = {
        'encoder_file_exists': False,
        'encoder_has_city': False,
        'encoder_city_classes': None,
        'model_features': None,
        'city_feature_index': None,
        'city_feature_type': None,
        'error': None
    }

    encoder_path = "label_encoders_final_deploy.pkl"
    feature_cols_path = "feature_columns_final_deploy.pkl"
    model_path = "best_model_final_deploy.pkl"

    # 1. 检查编码器文件
    if os.path.exists(encoder_path):
        result['encoder_file_exists'] = True
        try:
            encoders = joblib.load(encoder_path)
            if '城市' in encoders:
                result['encoder_has_city'] = True
                result['encoder_city_classes'] = encoders['城市'].classes_.tolist()
        except Exception as e:
            result['error'] = f"加载编码器失败: {e}"
            return result
    else:
        result['error'] = "编码器文件不存在。"

    # 2. 加载模型和特征列
    if os.path.exists(model_path) and os.path.exists(feature_cols_path):
        try:
            model = joblib.load(model_path)
            feature_cols = joblib.load(feature_cols_path)
            result['model_features'] = feature_cols
            if '城市' in feature_cols:
                city_idx = feature_cols.index('城市')
                result['city_feature_index'] = city_idx
                booster = model.get_booster()
                feature_types = booster.feature_types
                if city_idx < len(feature_types):
                    result['city_feature_type'] = feature_types[city_idx]
        except Exception as e:
            if result['error'] is None:
                result['error'] = f"加载模型或特征列失败: {e}"
    else:
        if result['error'] is None:
            result['error'] = "模型文件或特征列文件不存在。"

    return result


def display_city_diagnostic():
    """
    在 Streamlit 中显示城市特征诊断结果。
    """
    st.markdown("<div class='section-title'>🔧 城市特征诊断工具</div>", unsafe_allow_html=True)
    st.markdown("""
    此工具用于检查模型训练时的城市类别信息，帮助排查“所有城市都无法预测”的问题。
    """)

    if st.button("开始诊断", key="btn_city_diag"):
        with st.spinner("正在分析模型和编码器..."):
            result = inspect_city_categories()

        if result['error']:
            st.error(f"❌ 诊断过程中出现错误：{result['error']}")
            return

        # 显示编码器信息
        st.markdown("#### 📁 编码器文件检查")
        if result['encoder_file_exists']:
            st.success("编码器文件存在。")
            if result['encoder_has_city']:
                st.success("编码器中包含 '城市' 键。")
                st.markdown("**编码器中的城市类别列表：**")
                st.write(result['encoder_city_classes'])
                st.info("预测时传入的城市名称必须完全匹配以上列表之一（包括大小写、空格等）。")
            else:
                st.warning("编码器中 **未找到** '城市' 键，说明城市可能作为原生类别特征直接使用。")
        else:
            st.error("编码器文件不存在，无法确认城市类别。")

        # 显示模型特征信息
        st.markdown("#### 🤖 模型特征检查")
        if result['model_features'] is not None:
            st.success("成功加载模型特征列。")
            if result['city_feature_index'] is not None:
                st.markdown(f"- 城市特征在模型中的索引：{result['city_feature_index']}")
                if result['city_feature_type'] == 'c':
                    st.info("城市列被识别为**分类特征**（categorical）。预测时必须传入模型训练时见过的字符串类别。")
                elif result['city_feature_type'] == 'q':
                    st.warning("城市列被识别为**数值特征**，这通常意味着训练时未将其作为类别处理，而是进行了编码或直接作为数值。")
                else:
                    st.warning(f"城市特征类型为 '{result['city_feature_type']}'，请检查训练代码。")
            else:
                st.error("特征列中找不到 '城市'，请检查 `feature_columns_final_deploy.pkl` 的内容。")
        else:
            st.error("模型或特征列文件加载失败。")

        # 综合建议
        st.markdown("#### 💡 解决建议")
        if result['encoder_has_city']:
            st.markdown("""
            - 若预测时传入的城市名称不在编码器类别列表中，请修改界面选项或更新编码器。
            - 若编码器列表包含您选择的城市，但模型仍报错，可能是模型训练时未正确使用该编码器，建议重新训练模型。
            """)
        elif result['city_feature_type'] == 'c':
            st.markdown("""
            - 城市作为原生类别特征，但编码器中无对应信息。这通常是因为训练时使用了 `enable_categorical=True` 且直接传入字符串。
            - 请确认训练数据中城市列的具体取值（例如是否有额外空格、全角/半角字符）。
            - 最稳妥的方法是重新训练模型，确保训练数据与预测代码使用的城市名称完全一致。
            """)
        else:
            st.markdown("""
            - 如果城市列被当作数值特征，而预测时传入字符串，则必然出错。
            - 请检查训练代码：是否使用了 `LabelEncoder` 对城市进行编码？如果使用了，应在预测时同样进行编码，而不是传递原始字符串。
            """)

# 在 app.py 的适当位置调用（例如在页面底部）
with st.expander("🛠️ 模型诊断工具", expanded=False):
    display_city_diagnostic()
