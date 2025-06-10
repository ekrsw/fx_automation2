# MT5統合ガイド

## 概要

FX自動売買システムでのMetaTrader 5 (MT5) 統合について説明します。現在、macOS環境ではモック実装を使用しており、実際のMT5接続にはWindows環境が必要です。

## 現在の実装状況

### ✅ 完了済み（macOS）
- MT5接続管理のモック実装
- データ取得機能のモック実装
- リアルタイム価格データ生成
- 履歴データ生成
- シンボル情報取得
- 接続ヘルスチェック機能
- 包括的なテストスイート

### 📋 次のステップ（Windows環境）
- 実際のMT5パッケージ統合
- リアルMT5アカウントでの接続テスト
- ライブデータ取得の検証
- 取引実行機能の実装

## Windows環境でのセットアップ

### 1. 前提条件
- Windows 10/11 (64-bit)
- Python 3.8以上
- MetaTrader 5ターミナル

### 2. MetaTrader 5のインストール
```bash
# 1. MT5ターミナルのダウンロード
# https://www.metaquotes.net/en/metatrader5 からダウンロード

# 2. デモアカウントの作成
# MT5ターミナルでデモアカウントを開設
# 推奨ブローカー: MetaQuotes Demo, XM Demo, など

# 3. MT5 Pythonパッケージのインストール
pip install MetaTrader5
```

### 3. プロジェクトのセットアップ
```bash
# 1. プロジェクトクローン
git clone <repository-url>
cd fx_automation2

# 2. 仮想環境作成
python -m venv venv
venv\Scripts\activate

# 3. 依存関係インストール
pip install -r requirements.txt
pip install MetaTrader5

# 4. 環境変数設定
# .env ファイルを作成
MT5_LOGIN=あなたのログイン番号
MT5_PASSWORD=あなたのパスワード
MT5_SERVER=あなたのサーバー名
```

### 4. 設定の更新

#### `app/config.py` の更新
```python
# 実際の認証情報に変更
mt5_login: Optional[int] = int(os.getenv('MT5_LOGIN')) if os.getenv('MT5_LOGIN') else None
mt5_password: Optional[str] = os.getenv('MT5_PASSWORD')
mt5_server: Optional[str] = os.getenv('MT5_SERVER')
```

#### 実装ファイルの更新
`app/mt5/connection.py` と `app/mt5/data_fetcher.py` でモックコードを実際のMT5 APIコールに置き換え

## 接続テストの実行

### macOS（現在 - モックモード）
```bash
# モック機能のテスト
python scripts/test_mt5_connection.py

# 直接テスト実行
python -c "
import asyncio
import sys
sys.path.insert(0, '.')
from app.mt5.test_connection import MT5ConnectionTester
asyncio.run(MT5ConnectionTester().run_connection_tests())
"
```

### Windows（実MT5環境）
```bash
# 1. MT5ターミナルを起動してログイン
# 2. テスト実行
python scripts/test_mt5_connection.py

# 3. 結果確認
# - mt5_test_report.txt
# - mt5_test_results.json
```

## 実装の移行手順

### Phase 1: 基本接続（1-2日）
1. **環境準備**
   - Windows環境のセットアップ
   - MT5デモアカウント取得
   - 認証情報の設定

2. **基本接続テスト**
   ```python
   # 基本的な接続確認
   import MetaTrader5 as mt5
   
   # 初期化
   if not mt5.initialize():
       print("MT5初期化失敗")
   
   # ログイン
   if not mt5.login(login, password, server):
       print("ログイン失敗")
   
   # ターミナル情報取得
   terminal_info = mt5.terminal_info()
   account_info = mt5.account_info()
   ```

3. **接続管理の実装**
   - `app/mt5/connection.py` の更新
   - エラーハンドリングの追加
   - 再接続ロジックの実装

### Phase 2: データ取得（2-3日）
1. **リアルタイムデータ**
   ```python
   # ティックデータ取得
   tick = mt5.symbol_info_tick("USDJPY")
   
   # シンボル情報取得
   symbol_info = mt5.symbol_info("USDJPY")
   ```

2. **履歴データ**
   ```python
   # OHLC データ取得
   rates = mt5.copy_rates_from_pos("USDJPY", mt5.TIMEFRAME_H1, 0, 100)
   
   # 指定期間のデータ取得
   rates = mt5.copy_rates_range("USDJPY", mt5.TIMEFRAME_H1, start_date, end_date)
   ```

3. **データ検証**
   - データ整合性チェック
   - 欠損データの処理
   - タイムゾーン調整

### Phase 3: 取引機能（3-5日）
1. **デモ取引テスト**
   ```python
   # 注文送信
   request = {
       "action": mt5.TRADE_ACTION_DEAL,
       "symbol": "USDJPY",
       "volume": 0.1,
       "type": mt5.ORDER_TYPE_BUY,
       "price": mt5.symbol_info_tick("USDJPY").ask,
   }
   result = mt5.order_send(request)
   ```

2. **ポジション管理**
   - オープンポジション取得
   - ポジション修正・決済
   - 注文履歴確認

## パフォーマンス考慮事項

### 最適化ポイント
1. **接続管理**
   - 接続プールの実装
   - 自動再接続機能
   - タイムアウト管理

2. **データ取得**
   - キャッシュ機能
   - 非同期処理
   - レート制限対応

3. **エラーハンドリング**
   - ネットワーク断絶対応
   - MT5エラーコード処理
   - ログ記録

## セキュリティ

### 認証情報管理
```python
# 環境変数での管理
import os
from dotenv import load_dotenv

load_dotenv()

MT5_LOGIN = os.getenv('MT5_LOGIN')
MT5_PASSWORD = os.getenv('MT5_PASSWORD')
MT5_SERVER = os.getenv('MT5_SERVER')
```

### 本番環境での注意事項
- 認証情報の暗号化
- アクセスログの記録
- 不正アクセス検知
- 定期的な認証情報変更

## トラブルシューティング

### よくある問題

1. **接続失敗**
   ```
   症状: mt5.login() が False を返す
   原因: 認証情報の間違い、サーバー接続問題
   対策: 認証情報確認、MT5ターミナルでの手動ログイン確認
   ```

2. **データ取得失敗**
   ```
   症状: copy_rates_* が空のデータを返す
   原因: シンボルが無効、時間範囲の問題
   対策: シンボル一覧確認、時間範囲調整
   ```

3. **注文失敗**
   ```
   症状: order_send() がエラーを返す
   原因: 不十分な証拠金、無効な価格
   対策: アカウント情報確認、価格検証
   ```

### デバッグ手順
1. MT5エラーコードの確認: `mt5.last_error()`
2. ログファイルの確認
3. ネットワーク接続の確認
4. MT5ターミナルの状態確認

## テスト戦略

### 単体テスト
- 各MT5機能の個別テスト
- モックデータでの動作確認
- エラーケースのテスト

### 統合テスト
- 実MT5環境での接続テスト
- データフロー全体のテスト
- パフォーマンステスト

### 本番前テスト
- デモアカウントでの長期稼働テスト
- 各種市場条件でのテスト
- フェイルオーバーテスト

## まとめ

現在のモック実装により、MT5統合の基盤は完成しています。Windows環境での実装移行は、段階的に行うことで安全かつ確実に実現できます。

**次のアクション項目:**
1. Windows環境の準備
2. MT5デモアカウントの取得
3. Phase 1の基本接続実装
4. 包括的なテストの実行