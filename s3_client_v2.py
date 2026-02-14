import boto3
import os
import logging
from botocore.exceptions import NoCredentialsError, ClientError
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("S3_Client_v2")

class S3HybridClient:
    """
    AWS S3 (本番環境) と MinIO (ローカル開発環境) 間のシームレスな切り替えをサポートします。
    """
    def __init__(self):
        self.bucket_name = os.getenv("S3_BUCKET_NAME", "geosmart-etl-bucket")
        self.region = os.getenv("AWS_REGION", "us-east-1")
        ak = os.getenv("AWS_ACCESS_KEY_ID")
        sk = os.getenv("AWS_SECRET_ACCESS_KEY")
        endpoint = os.getenv("AWS_ENDPOINT_URL") # AWS と MinIO の主な違い

        self.s3 = None

        if ak and sk:
            try:
                
                # エンドポイントに値がある場合 (http://127.0.0.1:9000 など)、Boto3 は MinIO に接続します
                self.s3 = boto3.client(
                    's3',
                    aws_access_key_id=ak,
                    aws_secret_access_key=sk,
                    region_name=self.region,
                    endpoint_url=endpoint 
                )
                logger.info(f"[*] S3 クライアントの初期化に成功しました (Endpoint: {endpoint if endpoint else 'AWS Cloud'})")
                
                # [MinIO 固有のロジック] バケットが存在するかどうかを確認します。存在しない場合は自動的に作成します
                self._ensure_bucket_exists()
                
            except Exception as e:
                logger.error(f"[!] S3 接続失敗: {str(e)}")
        else:
            logger.warning("[!] キーが検出されませんでした。モック モードで実行されています。")

    def _ensure_bucket_exists(self):
        """バケットを自動的にチェックして作成する（ローカル MinIO 環境用）"""
        if not self.s3: return
        try:
            self.s3.head_bucket(Bucket=self.bucket_name)
        except ClientError:
            logger.info(f"[*] Bucket {self.bucket_name} 存在しません。自動的に作成します...")
            try:
                self.s3.create_bucket(Bucket=self.bucket_name)
                logger.info(f"[+] Bucket {self.bucket_name} 作成成功")
            except Exception as e:
                logger.error(f"[!] Bucket 作成失敗: {str(e)}")

    def upload_file(self, file_path: str, object_name: str = None):
        if object_name is None:
            object_name = os.path.basename(file_path)

        if self.s3:
            try:
                self.s3.upload_file(file_path, self.bucket_name, object_name)
                
                # アクセスリンクを生成する（MinIO localhost に適合）
                endpoint = os.getenv("AWS_ENDPOINT_URL")
                if endpoint:
                    url = f"{endpoint}/{self.bucket_name}/{object_name}"
                else:
                    url = f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{object_name}"
                
                logger.info(f"[+] アップロード成功: {url}")
                return {"status": "success", "url": url, "provider": "MinIO" if endpoint else "AWS"}
            except Exception as e:
                logger.error(f"[-] アップロード異常: {str(e)}")
                return {"status": "error", "detail": str(e)}
        else:
            # モックモード
            return {"status": "success", "url": "mock://upload", "provider": "Mock"}