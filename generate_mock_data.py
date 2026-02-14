import pandas as pd
import numpy as np

def generate_mock_dataset(filepath="forest_dataset.csv", num_rows=1000):
    """
    生成符合 GeoSmart-ETL 格式要求的模拟数据
    用于验证 data_loader_v1.py 的逻辑是否跑通
    """
    print(f"[*] 正在生成 {num_rows} 条模拟高维数据...")

    # 1. 生成 10 维连续变量 (随机浮点数)
    data = {
        'Elevation': np.random.normal(2500, 100, num_rows),
        'Aspect': np.random.uniform(0, 360, num_rows),
        'Slope': np.random.uniform(0, 45, num_rows),
        'Horizontal_Distance_To_Hydrology': np.random.uniform(0, 500, num_rows),
        'Vertical_Distance_To_Hydrology': np.random.uniform(-50, 200, num_rows),
        'Horizontal_Distance_To_Roadways': np.random.uniform(0, 5000, num_rows),
        'Hillshade_9am': np.random.randint(0, 255, num_rows),
        'Hillshade_Noon': np.random.randint(0, 255, num_rows),
        'Hillshade_3pm': np.random.randint(0, 255, num_rows),
        'Horizontal_Distance_To_Fire_Points': np.random.uniform(0, 5000, num_rows)
    }

    # 2. 生成 Wilderness_Area (4维，One-Hot 保证每行只有一个 1)
    # 逻辑：随机生成 0-3 的索引，然后转为 One-Hot
    wild_indices = np.random.randint(0, 4, num_rows)
    wild_one_hot = np.eye(4)[wild_indices]
    for i in range(4):
        data[f'Wilderness_Area{i+1}'] = wild_one_hot[:, i].astype(int)

    # 3. 生成 Soil_Type (40维，随机生成)
    # 为了测试简便，我们让大部分为 0，随机选一列为 1 (严格 One-Hot)
    soil_indices = np.random.randint(0, 40, num_rows)
    soil_one_hot = np.eye(40)[soil_indices]
    for i in range(40):
        data[f'Soil_Type{i+1}'] = soil_one_hot[:, i].astype(int)

    # 4. 生成 Label (Cover_Type 1-7)
    data['Cover_Type'] = np.random.randint(1, 8, num_rows)

    # 5. 构建 DataFrame 并保存
    df = pd.DataFrame(data)
    
    # 故意制造几个“极端值”来测试标准化效果 (可选)
    df.loc[0, 'Elevation'] = 3500 # 极高海拔
    
    df.to_csv(filepath, index=False)
    print(f"[+] 模拟数据已保存至: {filepath} (大小: {df.memory_usage().sum()/1024:.2f} KB)")

if __name__ == "__main__":
    generate_mock_dataset()