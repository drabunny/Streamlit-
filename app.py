import pandas as pd
import joblib
import os

# ================== 配置区域 ==================
# 请根据实际情况修改以下路径
TRAIN_DATA_PATH = "train_data.csv"                # 训练数据文件路径，支持 .csv 或 .pkl
ENCODER_PATH = "label_encoders_final_deploy.pkl"  # 编码器文件路径（可选，如果没有可设为 None）
EXPECTED_CITIES = ["济南市", "烟台市", "济宁市"]   # 预期出现的城市
# ============================================

def load_data(path):
    """根据文件扩展名加载数据"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在：{path}")
    if path.endswith('.csv'):
        return pd.read_csv(path)
    elif path.endswith('.pkl') or path.endswith('.pickle'):
        return pd.read_pickle(path)
    else:
        raise ValueError("不支持的文件格式，仅支持 CSV 或 Pickle")

def main():
    print("=" * 70)
    print("🔍 训练数据城市列检查工具")
    print("=" * 70)

    # 1. 加载训练数据
    print(f"\n📂 训练数据文件：{TRAIN_DATA_PATH}")
    try:
        df = load_data(TRAIN_DATA_PATH)
        print(f"✅ 成功加载数据，形状：{df.shape}")
    except Exception as e:
        print(f"❌ 加载数据失败：{e}")
        return

    # 2. 检查城市列是否存在
    city_col = '城市'
    if city_col not in df.columns:
        print(f"\n❌ 训练数据中不存在列 '{city_col}'。")
        print("现有列名：", df.columns.tolist())
        return
    print(f"\n✅ 找到列 '{city_col}'，数据类型：{df[city_col].dtype}")

    # 3. 缺失值检查
    null_count = df[city_col].isnull().sum()
    if null_count > 0:
        print(f"\n⚠️  城市列存在 {null_count} 个缺失值。")
    else:
        print("\n✅ 城市列无缺失值。")

    # 4. 唯一取值及样本数量
    value_counts = df[city_col].value_counts(dropna=False)
    print("\n📊 城市列取值分布：")
    print(value_counts.to_string())

    # 5. 检查预期城市
    unique_cities = df[city_col].astype(str).unique().tolist()
    print("\n🔎 预期城市检查：")
    all_present = True
    for city in EXPECTED_CITIES:
        if city in unique_cities:
            count = value_counts.get(city, 0)
            print(f"   ✅ 城市 '{city}' 存在，样本数：{count}")
        else:
            print(f"   ❌ 城市 '{city}' 不存在于训练数据中！")
            all_present = False

    if not all_present:
        print("\n⚠️  警告：训练数据缺少部分预期城市，模型将无法对这些城市进行预测。")
        print("   请补充数据并重新训练模型。")
    else:
        print("\n✅ 所有预期城市均已包含在训练数据中。")

    # 6. 额外检查：城市值是否为数字（可能已被编码）
    if df[city_col].dtype in ['int64', 'float64']:
        print("\n⚠️  注意：城市列是数值类型，可能已使用 LabelEncoder 编码。")
        print("   请确认预测时是否正确使用了相同的编码器。")
    elif df[city_col].dtype == 'object':
        print("\n✅ 城市列是文本类型，可直接作为类别特征使用。")

    # 7. 如果有编码器文件，检查编码器中的城市类别
    if ENCODER_PATH and os.path.exists(ENCODER_PATH):
        print(f"\n🔧 编码器文件：{ENCODER_PATH}")
        try:
            encoders = joblib.load(ENCODER_PATH)
            if '城市' in encoders:
                classes = encoders['城市'].classes_
                print(f"   编码器 '城市' 中的类别（共 {len(classes)} 个）：")
                for i, cls in enumerate(classes):
                    print(f"     {i}: {cls}")
                # 检查是否与训练数据一致
                data_cities = set(df[city_col].astype(str).unique())
                encoder_cities = set(classes)
                if data_cities != encoder_cities:
                    print("\n⚠️  训练数据中的城市值与编码器中的类别不一致！")
                    print(f"   仅存在于训练数据：{data_cities - encoder_cities}")
                    print(f"   仅存在于编码器：{encoder_cities - data_cities}")
                else:
                    print("\n✅ 训练数据与编码器类别一致。")
            else:
                print("   未找到 '城市' 键，可能特征名不同或未保存该编码器。")
        except Exception as e:
            print(f"   加载编码器失败：{e}")
    elif ENCODER_PATH and not os.path.exists(ENCODER_PATH):
        print(f"\n⚠️  编码器文件不存在：{ENCODER_PATH}")

    print("\n" + "=" * 70)
    print("检查完成")
    print("=" * 70)

if __name__ == "__main__":
    main()
