import pandas as pd
import numpy as np
import json
import os
from sklearn.preprocessing import StandardScaler

class ForestDataProcessor:
    """
    GeoSmart-ETL 核心数据处理类 (v2 - Runnable)
    
    变更日志:
    v1: 基础类结构与逻辑验证 (Dry Run)
    v2: 解除 process 调用限制，正式执行 ETL 流程
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
        print(f"[*] [Step 1] 正在加载数据: {self.raw_path} ...")
        
        if not os.path.exists(self.raw_path):
            raise FileNotFoundError(f"错误：找不到文件 {self.raw_path}，请确保已运行 generate_mock_data.py。")

        # 构建类型映射字典
        dtype_map = {col: 'int8' for col in self.binary_cols}
        dtype_map['Cover_Type'] = 'int8'
        for col in self.continuous_cols:
            dtype_map[col] = 'float32' # 浮点数降级

        # 加载数据
        self.df = pd.read_csv(self.raw_path, dtype=dtype_map)
        
        # 验证内存占用
        mem_usage = self.df.memory_usage(deep=True).sum() / 1024 ** 2
        print(f"    -> 数据加载完成。当前内存占用: {mem_usage:.2f} MB (已优化)")

    def _fold_categorical_features(self):
        """
        特征折叠 (Dimensionality Folding) - 为 AI Agent 准备
        逻辑：将 40 列 Soil_Type (One-Hot) 逆向解码为 1 列 Soil_Index
        """
        # 利用 idxmax 找到每一行中值为 1 的列名，然后提取数字部分
        soil_series = self.df[self.soil_cols].idxmax(axis=1).apply(lambda x: int(x.replace('Soil_Type', '')))
        return soil_series

    def validate_integrity(self):
        """
        数据熔断机制 (Circuit Breaker)
        """
        print("[*] [Step 2] 正在执行数据完整性检查...")
        
        # 1. 空值检查
        if self.df.isnull().sum().sum() > 0:
            raise ValueError("    -> 错误：数据中包含空值，清洗流程终止！")
        
        # 2. 逻辑一致性：每个样本必须且只能属于 1 个荒野区
        wilderness_sum = self.df[self.wilderness_cols].sum(axis=1)
        if not (wilderness_sum == 1).all():
            raise ValueError("    -> 错误：数据逻辑异常（荒野区 One-Hot 校验失败）。")

        print("    -> 数据完整性校验通过 (Circuit Breaker Passed)。")

    def process(self, output_parquet_path: str, output_profile_path: str):
        """
        主执行流
        """
        self.load_and_optimize()
        self.validate_integrity()

        # --- 特征工程 ---
        # 1. 连续变量标准化
        print("[*] [Step 3] 执行 Z-Score 标准化 (10维连续特征)...")
        self.df[self.continuous_cols] = self.scaler.fit_transform(self.df[self.continuous_cols])

        # 2. 生成 AI 摘要 (Profile)
        print("[*] [Step 4] 生成 AI 业务摘要 (Profile)...")
        soil_distribution = self._fold_categorical_features().value_counts(normalize=True).head(5).to_dict()
        
        # 将 numpy 类型转换为原生 python 类型，确保 JSON 序列化兼容
        profile = {
            "dataset_rows": int(len(self.df)),
            "cover_type_balance": {k: float(v) for k, v in self.df['Cover_Type'].value_counts(normalize=True).items()},
            "top_5_soil_types": {str(k): float(v) for k, v in soil_distribution.items()},
            "elevation_mean": float(self.df['Elevation'].mean()),
            "elevation_std": float(self.df['Elevation'].std())
        }

        # --- 持久化存储 ---
        # 1. 保存清洗后的 Parquet (高效二进制格式)
        self.df.to_parquet(output_parquet_path, index=False)
        print(f"    -> [Output] 清洗后数据已保存至: {output_parquet_path}")

        # 2. 保存 JSON 摘要 (供 LLM 读取)
        with open(output_profile_path, 'w') as f:
            json.dump(profile, f, indent=4)
        print(f"    -> [Output] AI 业务摘要已保存至: {output_profile_path}")

# --- 执行入口 ---
if __name__ == "__main__":
    # 实例化并运行
    processor = ForestDataProcessor(raw_path="forest_dataset.csv")
    
    # 真正的执行逻辑 (不再注释)
    processor.process("cleaned_forest_data.parquet", "data_profile.json")
    
    print("\n[SUCCESS] 阶段一 (ETL) 任务全部完成！准备进入阶段二 (FastAPI)。")