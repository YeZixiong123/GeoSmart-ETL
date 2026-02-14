GeoSmart-ETL: 高次元地理空間データ処理と AI インサイトプラットフォーム

📌 プロジェクトのポジショニング

GeoSmart-ETL は、クラウドネイティブアーキテクチャ向けの高性能データエンジニアリングプラットフォームです。大規模な高次元地理空間データ（Kaggle Forest Cover Typeを例に採用）を対象に設計されており、メモリ最適化ETL、S3互換ストレージ、そしてAIによるインテリジェント分析までのエンドツーエンドのサイクルを実現しています。

シニアデータエンジニアの視点から、本プロジェクトは 「高次元疎データにおけるメモリボトルネック」 と 「AI推論時のトークンコスト最適化」 という2つの大きな業界課題を解決しています。

🚀 核心的な技術ハイライト

1. 徹底したメモリ最適化 (Memory Efficiency)

型ダウンキャスト技術: 54次元の特徴量に対し、明示的な dtype マッピングを用いて44次元のバイナリ特徴量を int64 から int8 へと強制的にダウンキャスト。

成果: 全58万行のデータを処理する際、メモリ使用量を 80% 以上削減することに成功。

2. クラウドネイティブ・ストレージ・アダプター (Cloud-Native Storage)

S3 ハイブリッド・クライアント: boto3 をベースにカプセル化し、ローカルの MinIO とクラウドの AWS S3 をシームレスに切り替え可能。

コンピュートとストレージの分離: 環境変数による endpoint_url の制御により、「ローカル開発、クラウドデプロイ」をコード変更ゼロで実現。

3. コスト意識型 AI アーキテクチャ (Summary-Driven RAG)

特徴量フォールディング (Feature Folding): 40列の疎な土壌特徴量を逆シリアル化してエンコード。

ロジカルサマリー抽出: AIが巨大な生ファイルではなく1KBのロジカルサマリーのみを読み取ることで、1回あたりの推論コストを 99% 削減。

🖥️ 動作デモ (Application Preview)

本项目の実装成果は以下の通りです：

1. データ分析ダッシュボード

58万行のデータを瞬時に処理し、標高やサンプルサイズを正確に抽出。



2. AI インサイトアシスタント

DeepSeek-V3 を活用し、地形データに基づいた植栽推奨ロジックを展開。
<img width="2405" height="1179" alt="image" src="https://github.com/user-attachments/assets/8ac1ba1f-10b9-4fe6-a5ed-fc87c9c7a689" />
<img width="1137" height="926" alt="image" src="https://github.com/user-attachments/assets/738f010b-7979-45af-81f0-c6ff7fe9b69a" />


🛠️ 技術スタック

バックエンド: FastAPI (Async I/O), Uvicorn

データ処理: Pandas (Optimized), Scikit-learn, PyArrow (Parquet)

インフラ: Docker, MinIO (S3 Compatible)

AI: DeepSeek-V3 (via OpenAI API SDK)

📂 クイックスタート

1. 環境構築

.env.example をコピーして .env にリネームし、APIキーを入力してください：

AI_API_KEY=your_key_here
AWS_ENDPOINT_URL=[http://127.0.0.1:9000](http://127.0.0.1:9000)


2. インフラの起動 (Docker)

docker run -d -p 9000:9000 -p 9001:9001 --name minio-s3 minio/minio server /data --console-address ":9001"


3. サービスの実行

pip install -r requirements.txt
python main.py


http://localhost:8000 にアクセスして利用を開始します。

📊 データ資産の説明

生データ: Kaggle Forest Cover Type Prediction より提供。

処理済み成果物:

*.parquet: 機械学習モデル向けの効率的な列指向ストレージ形式。

*_profile.json: AIエージェントが消費するためのビジネスメタデータ。

⚖️ License

Distributed under the MIT License. See LICENSE for more information.
