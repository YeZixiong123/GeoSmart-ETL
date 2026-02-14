import pandas as pd
import numpy as np
import json
import os
from sklearn.preprocessing import StandardScaler

class ForestDataProcessor:
    def __init__(self, raw_path: str):
        self.raw_path = raw_path
        self.df = None
        self.scaler = StandardScaler()
        
        self.continuous_cols = [
            'Elevation', 'Aspect', 'Slope', 
            'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
            'Horizontal_Distance_To_Roadways', 
            'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 
            'Horizontal_Distance_To_Fire_Points'
        ]
        
        self.wilderness_cols = [f'Wilderness_Area{i}' for i in range(1, 5)]
        self.soil_cols = [f'Soil_Type{i}' for i in range(1, 41)]
        self.binary_cols = self.wilderness_cols + self.soil_cols

    def load_and_optimize(self):
        print(f"[*] [Step 1] データの読み込み中: {self.raw_path} ...")
        dtype_map = {col: 'int8' for col in self.binary_cols}
        if 'Cover_Type' in pd.read_csv(self.raw_path, nrows=1).columns:
            dtype_map['Cover_Type'] = 'int8'
        for col in self.continuous_cols:
            dtype_map[col] = 'float32'

        self.df = pd.read_csv(self.raw_path, dtype=dtype_map)
        print(f"    -> 行数: {len(self.df)}")

    def validate_integrity(self):
        print("[*] [Step 2] データの完全性を検証中...")
        if self.df.isnull().sum().sum() > 0:
            raise ValueError("データに欠損値が含まれています")
        print("    -> 検証完了")

    def process(self, output_parquet_path: str, output_profile_path: str):
        self.load_and_optimize()
        self.validate_integrity()

        print("[*] [Step 3] 生成中")
        
        # 
        raw_elevation_mean = float(self.df['Elevation'].mean())
        raw_elevation_std = float(self.df['Elevation'].std())
        
        # 土壌分布のトップ5を抽出
        soil_series = self.df[self.soil_cols].idxmax(axis=1).apply(lambda x: int(x.replace('Soil_Type', '')))
        soil_distribution = soil_series.value_counts(normalize=True).head(5).to_dict()

        profile = {
            "dataset_rows": int(len(self.df)),
            "elevation_mean": raw_elevation_mean, 
            "elevation_std": raw_elevation_std,
            "top_5_soil_types": {str(k): float(v) for k, v in soil_distribution.items()}
        }

        # 機械学習の前に、連続変数を標準化
        print("[*] [Step 4] 標準化中...")
        self.df[self.continuous_cols] = self.scaler.fit_transform(self.df[self.continuous_cols])

        # --- 持久化 ---
        self.df.to_parquet(output_parquet_path, index=False)
        with open(output_profile_path, 'w') as f:
            json.dump(profile, f, indent=4)
        
        print(f"    -> [DONE] 平均値とる: {raw_elevation_mean:.2f} m")

if __name__ == "__main__":

    processor = ForestDataProcessor(raw_path="forest_dataset.csv")
    processor.process("processed/train_cleaned.parquet", "processed/train_profile.json")