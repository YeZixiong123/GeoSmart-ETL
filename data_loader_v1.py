import pandas as pd
import numpy as np
import json
import os
from sklearn.preprocessing import StandardScaler

class ForestDataProcessor:
    """
    GeoSmart-ETL 核心数据处理类 (v1)
    
    职责：
    1. 内存优化摄入 (Memory Optimized Ingestion)
    2. 特征折叠与元数据提取 (Feature Folding & Profiling)
    3. 数据清洗与标准化 (Cleaning & Normalization)
    """

    def __init__(self, raw_path: str):
        self.raw_path = raw_path
        self.df = None
        self.scaler = StandardScaler()
        
        # --- 1. 定义 Schema (核心工程化配置) ---
        # 连续变量 (10维) - 将用于标准化
        self.continuous_cols = [
            'Elevation', 'Aspect', 'Slope', 
            'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
            'Horizontal_Distance_To_Roadways', 
            'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 
            'Horizontal_Distance_To_Fire_Points'
        ]
        
        # 二值特征 (44维) - 将用于折叠以节省 Token
        # 动态生成列名：Wilderness_Area1~4, Soil_Type1~40
        self.wilderness_cols = [f'Wilderness_Area{i}' for i in range(1, 5)]
        self.soil_cols = [f'Soil_Type{i}' for i in range(1, 41)]
        self.binary_cols = self.wilderness_cols + self.soil_cols

    def load_and_optimize(self):
        """
        阶段一核心逻辑：内存优化加载
        策略：显式指定 dtype，避免 Pandas 默认将 0/1 识别为 int64 (8字节)，强制降级为 int8 (1字节)
        """
        print(f"[*] 正在加载数据: {self.raw_path} ...")
        
        if not os.path.exists(self.raw_path):
            raise FileNotFoundError(f"错误：找不到文件 {self.raw_path}，请确保数据在 input 目录下。")

        # 构建类型映射字典
        dtype_map = {col: 'int8' for col in self.binary_cols}
        dtype_map['Cover_Type'] = 'int8'
        for col in self.continuous_cols:
            dtype_map[col] = 'float32' # 浮点数降级

        # 加载数据
        self.df = pd.read_csv(self.raw_path, dtype=dtype_map)
        
        # 验证内存占用
        mem_usage = self.df.memory_usage(deep=True).sum() / 1024 ** 2
        print(f"[+] 数据加载完成。当前内存占用: {mem_usage:.2f} MB")

    def _fold_categorical_features(self):
        """
        特征折叠 (Dimensionality Folding) - 为 AI Agent 准备
        逻辑：将 40 列 Soil_Type (One-Hot) 逆向解码为 1 列 Soil_Index
        目的：后续给 LLM 看数据分布时，不用发送 40 个稀疏列，节省 Token。
        """
        # 利用 idxmax 找到每一行中值为 1 的列名，然后提取数字部分
        # 例如: 'Soil_Type29' -> 29
        soil_series = self.df[self.soil_cols].idxmax(axis=1).apply(lambda x: int(x.replace('Soil_Type', '')))
        return soil_series

    def validate_integrity(self):
        """
        数据熔断机制 (Circuit Breaker)
        """
        print("[*] 正在执行数据完整性检查...")
        
        # 1. 空值检查
        if self.df.isnull().sum().sum() > 0:
            raise ValueError("数据中包含空值，清洗流程终止！")
        
        # 2. 逻辑一致性：每个样本必须且只能属于 1 个荒野区
        wilderness_sum = self.df[self.wilderness_cols].sum(axis=1)
        if not (wilderness_sum == 1).all():
            raise ValueError("数据逻辑错误：存在样本不属于任何荒野区或属于多个荒野区。")

        print("[+] 数据完整性校验通过。")

    def process(self, output_parquet_path: str, output_profile_path: str):
        """
        主执行流
        """
        self.load_and_optimize()
        self.validate_integrity()

        # --- 特征工程 ---
        # 1. 连续变量标准化
        print("[*] 执行 Z-Score 标准化...")
        self.df[self.continuous_cols] = self.scaler.fit_transform(self.df[self.continuous_cols])

        # 2. 生成 AI 摘要 (Profile)
        print("[*] 生成 AI 业务摘要...")
        soil_distribution = self._fold_categorical_features().value_counts(normalize=True).head(5).to_dict()
        
        profile = {
            "dataset_rows": len(self.df),
            "cover_type_balance": self.df['Cover_Type'].value_counts(normalize=True).to_dict(),
            "top_5_soil_types": soil_distribution, # 只保留前5大土壤类型，节省 Token
            "elevation_mean": float(self.df['Elevation'].mean()), # 必须转为 python float 否则 json 报错
            "elevation_std": float(self.df['Elevation'].std())
        }

        # --- 持久化存储 ---
        # 1. 保存清洗后的 Parquet (高效二进制格式)
        self.df.to_parquet(output_parquet_path, index=False)
        print(f"[+] 清洗后数据已保存至: {output_parquet_path}")

        # 2. 保存 JSON 摘要 (供 LLM 读取)
        with open(output_profile_path, 'w') as f:
            json.dump(profile, f, indent=4)
        print(f"[+] AI 业务摘要已保存至: {output_profile_path}")

# --- 单元测试入口 ---
if __name__ == "__main__":
    # 模拟路径，假设用户会在项目根目录下创建一个 input 文件夹放 csv
    # 注意：你需要去 Kaggle 下载 train.csv 并重命名为 forest_dataset.csv
    processor = ForestDataProcessor(raw_path="forest_dataset.csv")
    
    # 由于还没有真实数据，这里仅仅为了测试代码语法是否正确，我们可以注释掉运行行
    # processor.process("cleaned_forest_data.parquet", "data_profile.json")
    print("代码编译通过！请下载数据集并在本地运行。")