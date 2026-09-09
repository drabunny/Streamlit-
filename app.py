import pandas as pd
import joblib

# ================== 请根据实际情况修改以下部分 ==================
# 假设训练数据保存在 CSV 文件中（如果是其他格式，请相应修改）
TRAIN_DATA_PATH = "train_data.csv"   # 修改为您的训练数据文件路径

# 如果需要检查保存的 LabelEncoder，请确保文件存在
ENCODER_PATH = "label_encoders_final_deploy.pkl"  # 编码器文件路径（可选）
# ================================================================

def main():
    print("=" * 60)
    print("开始检查训练数据中的城市列取值")
    print("=" * 60)

    # 1. 加载训练数据
    try:
        df = pd.read_csv(TRAIN_DATA_PATH)
        print(f"\n✅ 成功加载训练数据：{TRAIN_DATA_PATH}")
        print(f"数据形状：{df.shape}")
    except FileNotFoundError:
        print(f"\n❌ 未找到训练数据文件：{TRAIN_DATA_PATH}")
        print("请检查文件路径或文件名是否正确。")
        return
    except Exception as e:
        print(f"\n❌ 加载训练数据时出错：{e}")
        return

    # 2. 检查“城市”列是否存在
    city_col = '城市'
    if city_col not in df.columns:
        print(f"\n❌ 训练数据中不存在列 '{city_col}'，请确认特征列名称是否正确。")
        print("现有列名如下：")
        print(df.columns.tolist())
        return

    print(f"\n✅ 找到列 '{city_col}'，数据类型：{df[city_col].dtype}")

    # 3. 查看唯一取值及数量
    print("\n--- 城市列唯一取值及样本数量 ---")
    value_counts = df[city_col].value_counts(dropna=False)
    print(value_counts)

    # 4. 检查缺失值
    null_count = df[city_col].isnull().sum()
    if null_count > 0:
        print(f"\n⚠️ 城市列存在 {null_count} 个缺失值，请处理。")
    else:
        print("\n✅ 城市列无缺失值。")

    # 5. 检查是否包含预期城市
    expected_cities = ["济南市", "烟台市", "济宁市"]
    print("\n--- 预期城市检查 ---")
    unique_cities = df[city_col].astype(str).unique()
    for city in expected_cities:
        if city in unique_cities:
            count = len(df[df[city_col].astype(str) == city])
            print(f"✅ 城市 '{city}' 存在，样本数：{count}")
        else:
            print(f"❌ 城市 '{city}' 不存在于训练数据中！")

    # 6. 如果存在编码器文件，检查编码器中的城市类别
    print("\n--- 编码器检查（可选）---")
    try:
        encoders = joblib.load(ENCODER_PATH)
        if '城市' in encoders:
            print(f"编码器 '城市' 中的类别：{encoders['城市'].classes_}")
            print("请确认这些类别是否与训练数据中的唯一值对应。")
        else:
            print("编码器文件中未找到 '城市' 键，可能是特征名称不同或未保存该编码器。")
    except FileNotFoundError:
        print(f"未找到编码器文件：{ENCODER_PATH}，跳过此检查。")
    except Exception as e:
        print(f"加载编码器时出错：{e}")

    print("\n" + "=" * 60)
    print("检查完成")
    print("=" * 60)

if __name__ == "__main__":
    main()
