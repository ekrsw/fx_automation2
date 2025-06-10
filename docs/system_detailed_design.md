# FX自動売買システム詳細設計書
## ダウ理論・エリオット波動戦略 with Python FastAPI + MT5

---

## 1. システム概要

### 1.1 アーキテクチャ
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI       │    │  Analysis Engine │    │     MT5         │
│  (REST API)     │◄──►│   (Core Logic)   │◄──►│  (Data/Trading) │
│  - 監視UI       │    │  - ダウ理論      │    │  - 価格取得     │
│  - 設定管理     │    │  - エリオット波動│    │  - 注文実行     │
│  - 取引履歴     │    │  - リスク管理    │    │  - ポジション   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │
         ▼                        ▼
┌─────────────────┐    ┌──────────────────┐
│   Web Frontend  │    │ SQLite Database  │
│ - ダッシュボード│    │  + SQLAlchemy    │
│ - チャート表示  │    │  - 価格データ    │
│ - パラメータ設定│    │  - 取引履歴      │
└─────────────────┘    └──────────────────┘
```

### 1.2 技術スタック
- **Backend**: Python 3.11+ / FastAPI / Uvicorn
- **Database**: SQLite + SQLAlchemy + Alembic
- **Trading**: MetaTrader5 API
- **Analysis**: Pandas / NumPy / TA-Lib / Numba
- **Async**: asyncio / APScheduler
- **Frontend**: HTML/CSS/JavaScript (Vanilla or Vue.js)

---

## 2. プロジェクト構造

```
fx_auto_trader/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI アプリケーション
│   ├── config.py               # 設定管理
│   ├── dependencies.py         # DI設定
│   │
│   ├── api/                    # REST API エンドポイント
│   │   ├── __init__.py
│   │   ├── dashboard.py        # ダッシュボード API
│   │   ├── trading.py          # 取引制御 API
│   │   ├── analysis.py         # 分析結果 API
│   │   └── settings.py         # 設定管理 API
│   │
│   ├── core/                   # 核となるビジネスロジック
│   │   ├── __init__.py
│   │   ├── dow_theory.py       # ダウ理論実装
│   │   ├── elliott_wave.py     # エリオット波動実装
│   │   ├── strategy.py         # 統合戦略
│   │   ├── risk_manager.py     # リスク管理
│   │   └── signal_generator.py # シグナル生成
│   │
│   ├── db/                     # データベース関連
│   │   ├── __init__.py
│   │   ├── database.py         # DB接続設定
│   │   ├── models/             # SQLAlchemy モデル
│   │   │   ├── __init__.py
│   │   │   ├── price_data.py
│   │   │   ├── trades.py
│   │   │   ├── signals.py
│   │   │   └── settings.py
│   │   └── crud/               # CRUD操作
│   │       ├── __init__.py
│   │       ├── price_data.py
│   │       ├── trades.py
│   │       └── signals.py
│   │
│   ├── mt5/                    # MT5関連
│   │   ├── __init__.py
│   │   ├── connection.py       # MT5接続管理
│   │   ├── data_fetcher.py     # データ取得
│   │   ├── order_manager.py    # 注文管理
│   │   └── position_manager.py # ポジション管理
│   │
│   ├── utils/                  # ユーティリティ
│   │   ├── __init__.py
│   │   ├── indicators.py       # テクニカル指標
│   │   ├── calculations.py     # 計算関数
│   │   ├── validators.py       # バリデーション
│   │   └── logger.py           # ログ管理
│   │
│   └── scheduler/              # スケジューラ
│       ├── __init__.py
│       ├── tasks.py            # 定期実行タスク
│       └── job_manager.py      # ジョブ管理
│
├── frontend/                   # フロントエンド
│   ├── static/
│   │   ├── css/
│   │   ├── js/
│   │   └── images/
│   └── templates/
│       ├── index.html
│       ├── dashboard.html
│       └── settings.html
│
├── tests/                      # テストコード
│   ├── __init__.py
│   ├── test_core/
│   ├── test_api/
│   └── test_mt5/
│
├── migrations/                 # Alembic マイグレーション
├── config/
│   ├── settings.yaml
│   └── trading_params.yaml
│
├── logs/                       # ログファイル
├── data/                       # データファイル
│   └── fx_trading.db          # SQLite データベース
│
├── requirements.txt
├── README.md
└── main.py                     # アプリケーション起動点
```

---

## 3. データベース設計

### 3.1 SQLAlchemy モデル

#### 3.1.1 価格データ (PriceData)
```python
class PriceData(Base):
    __tablename__ = "price_data"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False)  # M1, M5, M15, M30, H1, H4, D1
    timestamp = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Integer, default=0)
    spread = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint('symbol', 'timeframe', 'timestamp'),
        Index('idx_symbol_timeframe_timestamp', 'symbol', 'timeframe', 'timestamp'),
    )
```

#### 3.1.2 取引記録 (Trade)
```python
class Trade(Base):
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True, index=True)
    ticket = Column(Integer, unique=True, index=True)  # MT5チケット番号
    symbol = Column(String(10), nullable=False)
    trade_type = Column(Enum(TradeType), nullable=False)  # BUY, SELL
    volume = Column(Float, nullable=False)
    open_price = Column(Float, nullable=False)
    close_price = Column(Float)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    open_time = Column(DateTime, nullable=False)
    close_time = Column(DateTime)
    profit = Column(Float, default=0.0)
    commission = Column(Float, default=0.0)
    swap = Column(Float, default=0.0)
    status = Column(Enum(TradeStatus), default=TradeStatus.OPEN)
    strategy_id = Column(String(50))  # どの戦略での取引か
    signal_id = Column(Integer, ForeignKey("signals.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
```

#### 3.1.3 シグナル記録 (Signal)
```python
class Signal(Base):
    __tablename__ = "signals"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), nullable=False)
    timeframe = Column(String(10), nullable=False)
    signal_type = Column(Enum(SignalType), nullable=False)  # BUY, SELL, CLOSE
    strategy = Column(String(50), nullable=False)  # DOW_ELLIOTT, DOW_ONLY, etc.
    confidence = Column(Float, nullable=False)  # 0.0-1.0
    entry_price = Column(Float)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    dow_trend = Column(String(20))  # UPTREND, DOWNTREND, SIDEWAYS
    elliott_wave = Column(String(10))  # Wave1, Wave2, etc.
    fibonacci_level = Column(Float)
    rr_ratio = Column(Float)  # Risk-Reward Ratio
    timestamp = Column(DateTime, nullable=False)
    executed = Column(Boolean, default=False)
    trade_id = Column(Integer, ForeignKey("trades.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
```

#### 3.1.4 システム設定 (SystemSettings)
```python
class SystemSettings(Base):
    __tablename__ = "system_settings"
    
    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(100), unique=True, nullable=False)
    value = Column(Text, nullable=False)
    value_type = Column(String(20), default="string")  # string, int, float, bool, json
    description = Column(Text)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
```

### 3.2 インデックス戦略
```sql
-- 高速クエリのためのインデックス
CREATE INDEX idx_price_symbol_timeframe_time ON price_data(symbol, timeframe, timestamp DESC);
CREATE INDEX idx_trades_symbol_status ON trades(symbol, status);
CREATE INDEX idx_signals_timestamp ON signals(timestamp DESC);
CREATE INDEX idx_signals_strategy_executed ON signals(strategy, executed);
```

---

## 4. MT5統合設計

### 4.1 接続管理 (connection.py)
```python
class MT5Connection:
    def __init__(self):
        self.connected = False
        self.last_heartbeat = None
        self.connection_retry_count = 0
        
    async def connect(self, login: int, password: str, server: str):
        """MT5接続を確立"""
        
    async def disconnect(self):
        """接続を閉じる"""
        
    async def health_check(self):
        """接続状態確認"""
        
    async def reconnect(self):
        """自動再接続"""
```

### 4.2 データ取得 (data_fetcher.py)
```python
class DataFetcher:
    def __init__(self, connection: MT5Connection):
        self.connection = connection
        
    async def get_live_prices(self, symbols: List[str]) -> Dict[str, Tick]:
        """リアルタイム価格取得"""
        
    async def get_ohlc_data(self, symbol: str, timeframe: int, count: int) -> pd.DataFrame:
        """OHLC履歴データ取得"""
        
    async def get_historical_data(self, symbol: str, timeframe: int, 
                                 start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """期間指定履歴データ取得"""
```

### 4.3 注文管理 (order_manager.py)
```python
class OrderManager:
    def __init__(self, connection: MT5Connection):
        self.connection = connection
        
    async def place_market_order(self, symbol: str, order_type: OrderType, 
                                volume: float, sl: float = None, tp: float = None) -> TradeResult:
        """成行注文"""
        
    async def place_pending_order(self, symbol: str, order_type: OrderType, 
                                 volume: float, price: float, sl: float = None, 
                                 tp: float = None) -> TradeResult:
        """指値・逆指値注文"""
        
    async def modify_position(self, ticket: int, sl: float = None, tp: float = None) -> bool:
        """ポジション修正"""
        
    async def close_position(self, ticket: int, volume: float = None) -> bool:
        """ポジション決済"""
```

---

## 5. 核心ロジック設計

### 5.1 ダウ理論実装 (dow_theory.py)

```python
@dataclass
class SwingPoint:
    timestamp: datetime
    price: float
    point_type: SwingPointType  # HIGH, LOW
    significance: float  # 重要度スコア

@dataclass
class TrendAnalysis:
    trend_direction: TrendDirection  # UPTREND, DOWNTREND, SIDEWAYS
    strength: float  # 0.0-1.0
    swing_points: List[SwingPoint]
    last_higher_high: SwingPoint
    last_higher_low: SwingPoint
    confirmation_score: float

class DowTheoryAnalyzer:
    def __init__(self, atr_multiplier: float = 0.5, swing_period: int = 5):
        self.atr_multiplier = atr_multiplier
        self.swing_period = swing_period
        
    def detect_swing_points(self, df: pd.DataFrame) -> List[SwingPoint]:
        """スイングポイント検出"""
        # ZigZag + ATRフィルタリング実装
        
    def analyze_trend(self, df: pd.DataFrame) -> TrendAnalysis:
        """トレンド分析"""
        # 高値・安値の切り上げ・切り下げ判定
        
    def calculate_trend_strength(self, swing_points: List[SwingPoint]) -> float:
        """トレンド強度計算"""
        
    def get_trend_confirmation(self, df: pd.DataFrame, volume_proxy: pd.Series) -> float:
        """出来高による確認（疑似出来高使用）"""
```

### 5.2 エリオット波動実装 (elliott_wave.py)

```python
@dataclass
class ElliottWave:
    wave_number: int  # 1-5 for impulse, A-C for correction
    wave_type: WaveType  # IMPULSE, CORRECTION
    start_point: SwingPoint
    end_point: SwingPoint
    sub_waves: List['ElliottWave']
    fibonacci_ratios: Dict[str, float]
    confidence: float

class ElliottWaveAnalyzer:
    def __init__(self, fibonacci_tolerance: float = 0.1):
        self.fibonacci_tolerance = fibonacci_tolerance
        self.fibonacci_ratios = {
            'wave_2_retracement': [0.382, 0.5, 0.618],
            'wave_3_extension': [1.618, 2.618],
            'wave_4_retracement': [0.236, 0.382],
            'wave_5_projection': [1.0, 1.618]
        }
        
    def identify_wave_pattern(self, swing_points: List[SwingPoint]) -> List[ElliottWave]:
        """波動パターン識別"""
        
    def validate_elliott_rules(self, waves: List[ElliottWave]) -> bool:
        """エリオット波動ルール検証"""
        # Wave 2 < Wave 1 start
        # Wave 3 != shortest
        # Wave 4 not overlap Wave 1
        
    def calculate_fibonacci_targets(self, waves: List[ElliottWave]) -> Dict[str, float]:
        """フィボナッチターゲット計算"""
        
    def predict_next_wave(self, current_waves: List[ElliottWave]) -> ElliottWave:
        """次の波動予測"""
```

### 5.3 統合戦略 (strategy.py)

```python
@dataclass
class StrategySignal:
    signal_type: SignalType
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    dow_analysis: TrendAnalysis
    elliott_analysis: List[ElliottWave]
    rationale: str

class DowElliottStrategy:
    def __init__(self, config: StrategyConfig):
        self.dow_analyzer = DowTheoryAnalyzer()
        self.elliott_analyzer = ElliottWaveAnalyzer()
        self.config = config
        
    async def analyze(self, symbol: str, timeframe: str) -> StrategySignal:
        """総合分析実行"""
        # 1. マルチタイムフレーム分析
        daily_data = await self.get_data(symbol, "D1", 252)
        h4_data = await self.get_data(symbol, "H4", 168)
        
        # 2. ダウ理論分析（日足）
        dow_trend = self.dow_analyzer.analyze_trend(daily_data)
        
        # 3. エリオット波動分析（4時間足）
        elliott_waves = self.elliott_analyzer.identify_wave_pattern(
            self.dow_analyzer.detect_swing_points(h4_data)
        )
        
        # 4. エントリー条件判定
        return self._generate_signal(dow_trend, elliott_waves, h4_data)
        
    def _generate_signal(self, dow_trend: TrendAnalysis, 
                        elliott_waves: List[ElliottWave],
                        current_data: pd.DataFrame) -> StrategySignal:
        """シグナル生成ロジック"""
        # 第3波開始条件チェック
        # リスクリワード計算
        # 信頼度スコア算出
```

---

## 6. API設計

### 6.1 エンドポイント一覧

#### 6.1.1 ダッシュボード API
```python
# GET /api/dashboard/status
# システム全体の状態取得

# GET /api/dashboard/performance
# パフォーマンス指標取得

# GET /api/dashboard/positions
# 現在のポジション一覧

# GET /api/dashboard/recent-trades
# 最近の取引履歴
```

#### 6.1.2 取引制御 API
```python
# POST /api/trading/start
# 自動売買開始

# POST /api/trading/stop
# 自動売買停止

# POST /api/trading/manual-trade
# 手動取引実行

# GET /api/trading/signals
# 現在のシグナル状況
```

#### 6.1.3 分析 API
```python
# GET /api/analysis/dow-trend/{symbol}
# ダウ理論分析結果

# GET /api/analysis/elliott-waves/{symbol}
# エリオット波動分析結果

# GET /api/analysis/backtest
# バックテスト実行・結果取得
```

### 6.2 WebSocket API
```python
# /ws/live-prices
# リアルタイム価格配信

# /ws/signals
# シグナル配信

# /ws/trades
# 取引実行通知
```

---

## 7. 非同期処理・スケジューラ設計

### 7.1 定期実行タスク
```python
class TradingScheduler:
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        
    async def start(self):
        # 価格データ取得（1分毎）
        self.scheduler.add_job(
            self.fetch_price_data,
            'interval',
            minutes=1,
            id='price_data_fetcher'
        )
        
        # 分析実行（5分毎）
        self.scheduler.add_job(
            self.run_analysis,
            'interval',
            minutes=5,
            id='strategy_analyzer'
        )
        
        # ヘルスチェック（30秒毎）
        self.scheduler.add_job(
            self.health_check,
            'interval',
            seconds=30,
            id='health_checker'
        )
```

### 7.2 イベント駆動処理
```python
class EventManager:
    def __init__(self):
        self.event_handlers = {}
        
    async def emit(self, event_type: str, data: dict):
        """イベント発火"""
        
    async def on_price_update(self, price_data: PriceData):
        """価格更新イベント"""
        
    async def on_signal_generated(self, signal: StrategySignal):
        """シグナル生成イベント"""
        
    async def on_trade_executed(self, trade: Trade):
        """取引実行イベント"""
```

---

## 8. 設定管理

### 8.1 設定ファイル構造
```yaml
# config/settings.yaml
mt5:
  login: 12345678
  password: "password"
  server: "MetaQuotes-Demo"
  timeout: 30

database:
  url: "sqlite:///./data/fx_trading.db"
  echo: false

trading:
  symbols: ["USDJPY", "EURJPY", "GBPJPY"]
  default_volume: 0.1
  max_positions: 3
  risk_per_trade: 0.02
  
strategy:
  dow_theory:
    atr_multiplier: 0.5
    swing_period: 5
    trend_confirmation_bars: 3
    
  elliott_wave:
    fibonacci_tolerance: 0.1
    min_wave_size: 0.001
    max_waves_to_analyze: 20
    
  risk_management:
    max_daily_loss: 0.05
    max_drawdown: 0.15
    position_sizing_method: "fixed_risk"

logging:
  level: "INFO"
  file: "logs/trading.log"
  max_size: "10MB"
  backup_count: 5
```

### 8.2 動的設定更新
```python
class ConfigManager:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self.load_config()
        
    async def update_config(self, key: str, value: Any):
        """設定更新（即座に反映）"""
        
    async def reload_config(self):
        """設定ファイル再読み込み"""
```

---

## 9. エラーハンドリング・ログ設計

### 9.1 カスタム例外
```python
class TradingSystemError(Exception):
    """基底例外クラス"""
    
class MT5ConnectionError(TradingSystemError):
    """MT5接続エラー"""
    
class InsufficientMarginError(TradingSystemError):
    """証拠金不足エラー"""
    
class InvalidSignalError(TradingSystemError):
    """無効なシグナルエラー"""
```

### 9.2 ログ設計
```python
# 構造化ログ
logger.info("trade_executed", extra={
    "trade_id": trade.id,
    "symbol": trade.symbol,
    "volume": trade.volume,
    "entry_price": trade.open_price,
    "strategy": "dow_elliott"
})

# パフォーマンスログ
logger.info("analysis_performance", extra={
    "symbol": "USDJPY",
    "execution_time_ms": 150,
    "swing_points_detected": 12,
    "elliott_waves_found": 3
})
```

---

## 10. セキュリティ・認証

### 10.1 API認証
```python
# JWT トークンベース認証
class AuthenticationService:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        
    async def create_access_token(self, user_id: str) -> str:
        """アクセストークン生成"""
        
    async def verify_token(self, token: str) -> dict:
        """トークン検証"""
```

### 10.2 設定暗号化
```python
class SecureConfig:
    def __init__(self, encryption_key: str):
        self.cipher = Fernet(encryption_key)
        
    def encrypt_sensitive_data(self, data: str) -> str:
        """機密データ暗号化"""
        
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """機密データ復号化"""
```

---

## 11. テスト戦略

### 11.1 テスト構造
```python
# tests/test_core/test_dow_theory.py
class TestDowTheoryAnalyzer:
    async def test_swing_point_detection(self):
        """スイングポイント検出テスト"""
        
    async def test_trend_analysis(self):
        """トレンド分析テスト"""

# tests/test_integration/test_strategy.py
class TestDowElliottStrategy:
    async def test_signal_generation(self):
        """シグナル生成統合テスト"""
```

### 11.2 バックテスト機能
```python
class Backtester:
    def __init__(self, strategy: DowElliottStrategy):
        self.strategy = strategy
        
    async def run_backtest(self, start_date: datetime, end_date: datetime,
                          initial_balance: float) -> BacktestResult:
        """バックテスト実行"""
        
    def calculate_metrics(self, trades: List[Trade]) -> PerformanceMetrics:
        """パフォーマンス指標計算"""
```

---

## 12. デプロイ・運用

### 12.1 起動スクリプト
```bash
#!/bin/bash
# start_trading_system.sh

# 仮想環境アクティベート
source venv/bin/activate

# データベースマイグレーション
alembic upgrade head

# MT5起動確認
python -c "import MetaTrader5 as mt5; print('MT5 OK' if mt5.initialize() else 'MT5 NG')"

# アプリケーション起動
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 12.2 監視・アラート
```python
class SystemMonitor:
    def __init__(self):
        self.health_checks = []
        
    async def check_mt5_connection(self) -> bool:
        """MT5接続監視"""
        
    async def check_database_health(self) -> bool:
        """データベース健全性監視"""
        
    async def check_memory_usage(self) -> float:
        """メモリ使用量監視"""
        
    async def send_alert(self, message: str, severity: str):
        """アラート送信（メール/Slack等）"""
```

---

## 13. パフォーマンス最適化

### 13.1 データ処理最適化
```python
# Numba JIT コンパイル
@njit(cache=True)
def calculate_zigzag_fast(high: np.ndarray, low: np.ndarray, 
                         deviation: float) -> np.ndarray:
    """高速ZigZag計算"""
    
# ベクトル化処理
def calculate_atr_vectorized(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ベクトル化ATR計算"""
```

### 13.2 メモリ管理
```python
class DataManager:
    def __init__(self, max_memory_mb: int = 500):
        self.max_memory_mb = max_memory_mb
        
    async def cleanup_old_data(self):
        """古いデータの削除"""
        
    async def optimize_dataframes(self, df: pd.DataFrame) -> pd.DataFrame:
        """データフレーム最適化"""
```

---

この詳細設計に基づいて実装を進めることで、堅牢で高性能なFX自動売買システムを構築できます。特に重要な点は、理論の実装精度とリアルタイム処理のバランスを取ることです。