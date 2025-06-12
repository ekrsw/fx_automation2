# FX Auto Trading System - API Endpoints Specification

## Overview

この文書は、FX自動売買システムの全APIエンドポイントの詳細仕様を定義しています。システムは25以上のRESTエンドポイントとWebSocketサポートを提供し、リアルタイム取引自動化を実現します。

## 基本情報

- **開発環境URL**: `http://localhost:8000`
- **APIベースパス**: `/api`
- **WebSocket**: `ws://localhost:8000/ws`
- **APIフレームワーク**: FastAPI
- **認証**: なし（開発モード）
- **レスポンス形式**: JSON

## エンドポイント構成

### 1. ダッシュボードAPI (`/api/dashboard`)

システム監視とステータス情報の取得

#### GET `/api/dashboard/status`
**説明**: システム全体の包括的な状態情報を取得
**認証**: 不要
**レスポンス**: 200 OK

```json
{
  "timestamp": "2024-01-01T12:00:00",
  "overall_status": "healthy|unhealthy",
  "components": {
    "mt5_connection": {
      "status": "healthy",
      "details": {
        "connected": true,
        "server": "XMTrading-MT5 3",
        "account": 12345,
        "balance": 10000.0
      }
    },
    "trading_engine": {
      "status": "healthy",
      "session_status": "active|paused|stopped",
      "signals_processed": 25,
      "session_start_time": "2024-01-01T09:00:00"
    },
    "order_manager": {
      "status": "active",
      "active_orders": 3,
      "success_rate": "95.2%",
      "last_order_time": "2024-01-01T11:45:00"
    },
    "position_manager": {
      "status": "active",
      "open_positions": 2,
      "total_pnl": 150.0,
      "largest_position_size": 0.15
    },
    "risk_manager": {
      "status": "active",
      "emergency_stop": false,
      "risk_score": 0.3,
      "portfolio_exposure": 0.08
    }
  },
  "configuration": {
    "trading_symbols": ["USDJPY", "EURJPY", "GBPJPY"],
    "max_positions": 3,
    "risk_per_trade": 0.02,
    "analysis_methods": ["dow_theory", "elliott_wave"]
  }
}
```

#### GET `/api/dashboard/health`
**説明**: システムのシンプルなヘルスチェック
**レスポンス**: 200 OK

```json
{
  "status": "healthy|degraded|unhealthy",
  "timestamp": "2024-01-01T12:00:00",
  "mt5_connected": true,
  "trading_active": true,
  "components_healthy": 5,
  "components_total": 5
}
```

#### GET `/api/dashboard/performance`
**説明**: 取引パフォーマンス指標の取得
**クエリパラメータ**:
- `period` (string, default="1d"): 1h, 1d, 1w, 1m
**レスポンス**: 200 OK

```json
{
  "timestamp": "2024-01-01T12:00:00",
  "period": "1d",
  "trading_performance": {
    "total_trades": 45,
    "winning_trades": 28,
    "losing_trades": 17,
    "win_rate": 0.622,
    "profit_loss": 1250.50,
    "realized_pnl": 1100.0,
    "unrealized_pnl": 150.50,
    "average_trade_duration": "2.5h",
    "best_trade": 85.0,
    "worst_trade": -35.0
  },
  "order_performance": {
    "total_orders": 52,
    "successful_orders": 50,
    "failed_orders": 2,
    "success_rate": 0.962,
    "average_slippage": 0.8,
    "average_execution_time": "45ms"
  },
  "risk_metrics": {
    "sharpe_ratio": 1.25,
    "max_drawdown": 0.08,
    "recovery_factor": 3.2,
    "profit_factor": 1.65
  }
}
```

#### GET `/api/dashboard/positions`
**説明**: 現在の未決済ポジション一覧
**クエリパラメータ**:
- `symbol` (string, optional): 通貨ペアフィルター
**レスポンス**: 200 OK

```json
{
  "timestamp": "2024-01-01T12:00:00",
  "positions": [
    {
      "position_id": "pos_001",
      "symbol": "USDJPY",
      "position_type": "BUY|SELL",
      "volume": 0.1,
      "entry_price": 150.25,
      "current_price": 150.45,
      "unrealized_pnl": 20.0,
      "stop_loss": 149.80,
      "take_profit": 151.00,
      "entry_time": "2024-01-01T10:30:00",
      "strategy": "unified_analysis",
      "confidence": 0.85,
      "commission": 2.5,
      "swap": -0.5,
      "duration_hours": 1.5
    }
  ],
  "summary": {
    "total_positions": 1,
    "total_exposure": 15045.0,
    "total_unrealized_pnl": 20.0,
    "margin_used": 150.45,
    "margin_free": 9849.55
  }
}
```

#### GET `/api/dashboard/recent-trades`
**説明**: 最近の取引履歴
**クエリパラメータ**:
- `limit` (int, default=50): 最大取引数
- `symbol` (string, optional): 通貨ペアフィルター
- `days` (int, default=7): 過去何日分

**レスポンス**: 200 OK

```json
{
  "timestamp": "2024-01-01T12:00:00",
  "trades": [
    {
      "trade_id": "trade_001",
      "symbol": "USDJPY",
      "position_type": "SELL",
      "volume": 0.1,
      "entry_price": 150.80,
      "close_price": 150.40,
      "realized_pnl": 40.0,
      "commission": 5.0,
      "swap": -1.0,
      "net_profit": 34.0,
      "entry_time": "2024-01-01T09:00:00",
      "close_time": "2024-01-01T11:30:00",
      "duration_hours": 2.5,
      "profit_loss": "profit",
      "close_reason": "take_profit",
      "strategy": "elliott_wave",
      "confidence": 0.75
    }
  ],
  "summary": {
    "total_trades": 45,
    "win_rate": 0.622,
    "total_realized_pnl": 1100.0,
    "average_trade_pnl": 24.44,
    "best_trade": 85.0,
    "worst_trade": -35.0
  }
}
```

#### GET `/api/dashboard/orders`
**説明**: アクティブな注文一覧
**レスポンス**: 200 OK

```json
{
  "timestamp": "2024-01-01T12:00:00",
  "orders": [
    {
      "order_id": "ord_001",
      "symbol": "EURJPY",
      "order_type": "BUY_LIMIT|SELL_LIMIT|BUY_STOP|SELL_STOP",
      "volume": 0.1,
      "price": 162.50,
      "current_price": 162.75,
      "stop_loss": 161.80,
      "take_profit": 163.50,
      "created_time": "2024-01-01T11:00:00",
      "expiration": "2024-01-01T17:00:00",
      "status": "pending|partially_filled|cancelled",
      "filled_volume": 0.0,
      "strategy": "dow_theory",
      "comment": "API order from unified analysis"
    }
  ],
  "summary": {
    "total_orders": 3,
    "pending_orders": 2,
    "partially_filled": 1
  }
}
```

#### GET `/api/dashboard/risk-status`
**説明**: リスク管理状況
**レスポンス**: 200 OK

```json
{
  "timestamp": "2024-01-01T12:00:00",
  "risk_status": {
    "overall_risk_score": 0.3,
    "risk_level": "low|medium|high|critical",
    "emergency_stop_active": false,
    "risk_alerts": []
  },
  "portfolio_risk": {
    "total_exposure": 45000.0,
    "position_count": 3,
    "concentration_risk": 0.3,
    "correlation_risk": 0.2,
    "largest_position_pct": 0.4,
    "margin_level": 6543.2
  },
  "limits": {
    "max_positions": 3,
    "max_risk_per_trade": 0.02,
    "max_portfolio_risk": 0.10,
    "max_drawdown": 0.20
  },
  "violations": [],
  "recommendations": [
    "Consider reducing position size on USDJPY",
    "Monitor correlation between EURJPY and GBPJPY"
  ]
}
```

#### GET `/api/dashboard/market-data`
**説明**: 現在の市場データ
**クエリパラメータ**:
- `symbols` (string, optional): カンマ区切りの通貨ペアリスト

**レスポンス**: 200 OK

```json
{
  "timestamp": "2024-01-01T12:00:00",
  "market_data": {
    "USDJPY": {
      "symbol": "USDJPY",
      "bid": 150.25,
      "ask": 150.27,
      "spread": 0.02,
      "last_price": 150.26,
      "change": 0.15,
      "change_percent": 0.10,
      "daily_high": 150.80,
      "daily_low": 149.90,
      "timestamp": "2024-01-01T12:00:00",
      "volume": 15420,
      "volatility": 0.012
    }
  },
  "symbols_count": 1,
  "connection_status": "connected",
  "last_update": "2024-01-01T12:00:00"
}
```

---

### 2. 取引制御API (`/api/trading`)

取引セッションと注文管理

#### POST `/api/trading/session/start`
**説明**: 取引セッション開始
**リクエストボディ**:

```json
{
  "mode": "demo|live|simulation",
  "symbols": ["USDJPY", "EURJPY"],
  "max_positions": 3,
  "risk_per_trade": 0.02
}
```

**レスポンス**: 200 OK

```json
{
  "success": true,
  "session_id": "session_123",
  "mode": "demo",
  "started_at": "2024-01-01T12:00:00",
  "message": "Trading session started in demo mode",
  "configuration": {
    "symbols": ["USDJPY", "EURJPY"],
    "max_positions": 3,
    "risk_per_trade": 0.02
  }
}
```

#### POST `/api/trading/session/stop`
**説明**: 取引セッション停止
**レスポンス**: 200 OK

```json
{
  "success": true,
  "session_id": "session_123",
  "stopped_at": "2024-01-01T15:00:00",
  "message": "Trading session stopped",
  "summary": {
    "duration_hours": 3.0,
    "signals_processed": 12,
    "trades_executed": 8,
    "net_pnl": 125.50
  }
}
```

#### POST `/api/trading/session/pause`
**説明**: 取引の一時停止
**レスポンス**: 200 OK

#### POST `/api/trading/session/resume`
**説明**: 取引の再開
**レスポンス**: 200 OK

#### POST `/api/trading/signals/process`
**説明**: 取引シグナルの処理
**リクエストボディ**:

```json
{
  "symbol": "USDJPY",
  "signal_type": "MARKET_BUY|MARKET_SELL|LIMIT_BUY|LIMIT_SELL",
  "entry_price": 150.25,
  "stop_loss": 149.80,
  "take_profit": 151.00,
  "take_profit_2": 151.50,
  "confidence": 0.85,
  "urgency": "IMMEDIATE|HIGH|MEDIUM|LOW|WATCH",
  "volume": 0.1,
  "strategy": "unified_analysis",
  "comment": "API signal from Elliott Wave analysis"
}
```

**レスポンス**: 200 OK

```json
{
  "success": true,
  "signal_id": "signal_123",
  "processed_at": "2024-01-01T12:00:00",
  "message": "Signal processed successfully",
  "action_taken": "order_created|order_queued|signal_rejected",
  "order_id": "ord_001",
  "execution_details": {
    "executed_volume": 0.1,
    "execution_price": 150.25,
    "slippage": 0.0
  }
}
```

#### POST `/api/trading/orders/create`
**説明**: 新規注文作成
**リクエストボディ**:

```json
{
  "symbol": "USDJPY",
  "order_type": "MARKET_BUY|MARKET_SELL|BUY_LIMIT|SELL_LIMIT|BUY_STOP|SELL_STOP",
  "volume": 0.1,
  "price": 150.25,
  "stop_loss": 149.80,
  "take_profit": 151.00,
  "expiration": "2024-01-01T17:00:00",
  "comment": "API order",
  "magic_number": 12345
}
```

**レスポンス**: 200 OK

```json
{
  "success": true,
  "order_id": "ord_001",
  "mt5_ticket": 12345,
  "executed_volume": 0.1,
  "execution_price": 150.25,
  "commission": 2.5,
  "executed_at": "2024-01-01T12:00:00",
  "status": "executed|pending|rejected",
  "message": "Order executed successfully"
}
```

#### DELETE `/api/trading/orders/{order_id}`
**説明**: 注文キャンセル
**パスパラメータ**: `order_id` (string)
**レスポンス**: 200 OK

```json
{
  "success": true,
  "order_id": "ord_001",
  "cancelled_at": "2024-01-01T12:00:00",
  "message": "Order cancelled successfully"
}
```

#### PUT `/api/trading/positions/{position_id}/modify`
**説明**: ポジション修正
**パスパラメータ**: `position_id` (string)
**リクエストボディ**:

```json
{
  "position_id": "pos_001",
  "modification_type": "stop_loss|take_profit|both",
  "new_stop_loss": 149.90,
  "new_take_profit": 151.20,
  "reason": "Risk adjustment due to market volatility"
}
```

**レスポンス**: 200 OK

```json
{
  "success": true,
  "position_id": "pos_001",
  "modified_at": "2024-01-01T12:00:00",
  "modifications": {
    "old_stop_loss": 149.80,
    "new_stop_loss": 149.90,
    "old_take_profit": 151.00,
    "new_take_profit": 151.20
  },
  "message": "Position modified successfully"
}
```

#### POST `/api/trading/positions/{position_id}/close`
**説明**: ポジション決済
**パスパラメータ**: `position_id` (string)
**リクエストボディ**:

```json
{
  "position_id": "pos_001",
  "volume": 0.05,
  "reason": "Manual close due to news event"
}
```

**レスポンス**: 200 OK

```json
{
  "success": true,
  "position_id": "pos_001",
  "closed_volume": 0.05,
  "remaining_volume": 0.05,
  "close_price": 150.45,
  "realized_pnl": 20.0,
  "commission": 2.5,
  "net_profit": 17.5,
  "closed_at": "2024-01-01T12:00:00",
  "message": "Position partially closed successfully"
}
```

#### POST `/api/trading/positions/close-all`
**説明**: 全ポジション決済
**クエリパラメータ**: 
- `symbol` (optional): 通貨ペアフィルター
**レスポンス**: 200 OK

```json
{
  "success": true,
  "closed_positions": 3,
  "total_realized_pnl": 85.50,
  "closed_at": "2024-01-01T12:00:00",
  "details": [
    {
      "position_id": "pos_001",
      "symbol": "USDJPY",
      "realized_pnl": 20.0
    }
  ]
}
```

#### POST `/api/trading/risk/emergency-stop`
**説明**: 緊急停止の発動
**リクエストボディ**:

```json
{
  "reason": "Market volatility exceeds threshold",
  "close_positions": true,
  "cancel_orders": true
}
```

**レスポンス**: 200 OK

```json
{
  "success": true,
  "emergency_stop_activated": true,
  "activated_at": "2024-01-01T12:00:00",
  "actions_taken": {
    "positions_closed": 3,
    "orders_cancelled": 2,
    "trading_suspended": true
  },
  "message": "Emergency stop activated successfully"
}
```

#### POST `/api/trading/risk/emergency-stop/deactivate`
**説明**: 緊急停止の解除
**レスポンス**: 200 OK

---

### 3. 分析API (`/api/analysis`)

テクニカル分析とシグナル生成

#### GET `/api/analysis/dow-theory/{symbol}`
**説明**: ダウ理論分析
**パスパラメータ**: `symbol` (string)
**クエリパラメータ**:
- `timeframe` (string, default="H1"): M1, M5, M15, H1, H4, D1, W1, MN1
- `periods` (int, default=100): 分析期間
- `sensitivity` (float, default=0.5): 感度設定

**レスポンス**: 200 OK

```json
{
  "timestamp": "2024-01-01T12:00:00",
  "symbol": "USDJPY",
  "timeframe": "H1",
  "periods_analyzed": 100,
  "analysis": {
    "primary_trend": "bullish|bearish|sideways",
    "secondary_trend": "bullish|bearish|sideways",
    "minor_trend": "bullish|bearish|sideways",
    "trend_strength": 0.8,
    "trend_confidence": 0.75,
    "confirmation_score": 0.85,
    "swing_points": [
      {
        "index": 50,
        "price": 150.25,
        "type": "high|low",
        "timestamp": "2024-01-01T11:00:00",
        "significance": 0.9
      }
    ],
    "trend_lines": [
      {
        "type": "support|resistance",
        "start_point": {"index": 10, "price": 149.80},
        "end_point": {"index": 80, "price": 150.50},
        "strength": 0.9,
        "touches": 3
      }
    ],
    "key_levels": {
      "support": [149.50, 149.80, 150.00],
      "resistance": [150.50, 150.80, 151.00]
    }
  },
  "signals": [
    {
      "signal_type": "BULLISH_CONTINUATION",
      "confidence": 0.8,
      "entry_price": 150.25,
      "target_prices": [150.80, 151.20],
      "stop_loss": 149.80
    }
  ],
  "metadata": {
    "analysis_time": "2024-01-01T12:00:00",
    "reliability_score": 0.85,
    "data_quality": "high",
    "noise_level": 0.1
  }
}
```

#### GET `/api/analysis/elliott-wave/{symbol}`
**説明**: エリオット波動分析
**パスパラメータ**: `symbol` (string)
**クエリパラメータ**: ダウ理論と同様

**レスポンス**: 200 OK

```json
{
  "timestamp": "2024-01-01T12:00:00",
  "symbol": "USDJPY",
  "timeframe": "H1",
  "analysis": {
    "complete_patterns": [
      {
        "pattern_id": "impulse_001",
        "pattern_type": "impulse|corrective",
        "degree": "primary|intermediate|minor|minute|minuette",
        "waves": [
          {
            "wave_number": 1,
            "start_price": 149.50,
            "end_price": 150.20,
            "start_time": "2024-01-01T09:00:00",
            "end_time": "2024-01-01T10:00:00"
          }
        ],
        "confidence": 0.85,
        "fibonacci_relationships": {
          "wave_2_retracement": 0.618,
          "wave_3_extension": 1.618,
          "wave_4_retracement": 0.382
        }
      }
    ],
    "incomplete_patterns": [
      {
        "pattern_id": "impulse_002",
        "current_wave": 4,
        "expected_completion": "2024-01-01T14:00:00",
        "predicted_targets": [151.00, 151.50]
      }
    ],
    "fibonacci_levels": {
      "retracements": {
        "23.6": 150.10,
        "38.2": 149.90,
        "50.0": 149.75,
        "61.8": 149.60,
        "78.6": 149.45
      },
      "extensions": {
        "100": 151.00,
        "127.2": 151.20,
        "161.8": 151.50,
        "200": 151.75
      }
    }
  },
  "predictions": [
    {
      "scenario": "primary",
      "probability": 0.7,
      "target_price": 151.00,
      "target_time": "2024-01-01T15:00:00",
      "risk_level": 149.50
    }
  ],
  "metadata": {
    "patterns_detected": 5,
    "average_confidence": 0.78
  }
}
```

#### GET `/api/analysis/unified/{symbol}`
**説明**: 統合分析（ダウ理論 + エリオット波動）
**パスパラメータ**: `symbol` (string)
**クエリパラメータ**: 
- 基本パラメータはダウ理論と同様
- `dow_weight` (float, default=0.4): ダウ理論の重み
- `elliott_weight` (float, default=0.6): エリオット波動の重み

**レスポンス**: 200 OK

```json
{
  "timestamp": "2024-01-01T12:00:00",
  "symbol": "USDJPY",
  "timeframe": "H1",
  "unified_analysis": {
    "combined_signal": "bullish|bearish|neutral",
    "combined_confidence": 0.8,
    "agreement_score": 0.75,
    "consensus_strength": "strong|moderate|weak",
    "dow_analysis": {
      "signal": "bullish",
      "confidence": 0.7,
      "weight": 0.4,
      "contribution": 0.28
    },
    "elliott_analysis": {
      "signal": "bullish",
      "confidence": 0.9,
      "weight": 0.6,
      "contribution": 0.54
    },
    "combined_targets": {
      "primary_target": 151.00,
      "secondary_target": 151.50,
      "extended_target": 152.00,
      "confidence_levels": [0.8, 0.6, 0.4]
    },
    "risk_levels": {
      "stop_loss": 149.50,
      "major_support": 149.00,
      "invalidation_level": 148.50
    }
  },
  "recommendations": {
    "action": "BUY|SELL|HOLD",
    "confidence_grade": "A|B|C|D",
    "risk_grade": "LOW|MEDIUM|HIGH",
    "entry_timing": "immediate|wait_for_confirmation|avoid",
    "position_size": "normal|reduced|increased",
    "holding_period": "short|medium|long"
  },
  "divergences": [
    {
      "type": "signal_divergence",
      "description": "Dow Theory suggests stronger bullish signal",
      "impact": "low|medium|high"
    }
  ]
}
```

#### POST `/api/analysis/multi-timeframe`
**説明**: マルチタイムフレーム分析
**リクエストボディ**:

```json
{
  "symbol": "USDJPY",
  "timeframes": ["D1", "H4", "H1", "M15"],
  "periods": 100,
  "analysis_type": "unified|dow_theory|elliott_wave"
}
```

**レスポンス**: 200 OK

```json
{
  "timestamp": "2024-01-01T12:00:00",
  "symbol": "USDJPY",
  "multi_timeframe_analysis": {
    "overall_alignment": 0.8,
    "dominant_trend": "bullish",
    "confidence": 0.85,
    "timeframe_analysis": {
      "D1": {
        "trend": "bullish",
        "strength": 0.9,
        "confidence": 0.8,
        "weight": 0.4
      },
      "H4": {
        "trend": "bullish",
        "strength": 0.7,
        "confidence": 0.75,
        "weight": 0.3
      },
      "H1": {
        "trend": "bullish",
        "strength": 0.8,
        "confidence": 0.9,
        "weight": 0.2
      },
      "M15": {
        "trend": "neutral",
        "strength": 0.5,
        "confidence": 0.6,
        "weight": 0.1
      }
    },
    "alignment_score": {
      "trend_alignment": 0.85,
      "strength_alignment": 0.75,
      "confidence_alignment": 0.80
    },
    "optimal_entry_timeframe": "H1",
    "monitoring_timeframes": ["D1", "H4"]
  }
}
```

#### GET `/api/analysis/signals/{symbol}`
**説明**: 取引シグナル生成
**パスパラメータ**: `symbol` (string)
**クエリパラメータ**:
- `timeframe` (string, default="H1")
- `min_confidence` (float, default=0.6)

**レスポンス**: 200 OK

```json
{
  "timestamp": "2024-01-01T12:00:00",
  "symbol": "USDJPY",
  "signal_generated": true,
  "signal": {
    "signal_id": "sig_001",
    "signal_type": "MARKET_BUY|LIMIT_BUY|MARKET_SELL|LIMIT_SELL",
    "urgency": "IMMEDIATE|HIGH|MEDIUM|LOW|WATCH",
    "entry_price": 150.25,
    "entry_condition": "market|limit_at_price",
    "stop_loss": 149.80,
    "take_profit_1": 151.00,
    "take_profit_2": 151.50,
    "take_profit_3": 152.00,
    "volume_recommendation": 0.1,
    "confidence": 0.85,
    "risk_reward_ratio": 2.5,
    "expected_duration": "4h",
    "strategy": "unified_analysis",
    "created_at": "2024-01-01T12:00:00",
    "valid_until": "2024-01-01T18:00:00"
  },
  "confidence_analysis": {
    "overall_confidence": 0.85,
    "confidence_grade": "A",
    "confidence_factors": {
      "technical_analysis": 0.8,
      "market_structure": 0.9,
      "timeframe_alignment": 0.85,
      "risk_reward": 0.8,
      "historical_performance": 0.75,
      "market_conditions": 0.85
    },
    "risk_factors": [
      "News event scheduled in 2 hours",
      "Lower timeframe showing consolidation"
    ]
  },
  "execution_recommendations": {
    "order_type": "market|limit",
    "execution_strategy": "immediate|split|iceberg",
    "monitoring_required": true,
    "exit_strategy": "trailing|fixed|manual"
  }
}
```

#### GET `/api/analysis/signals/{symbol}/history`
**説明**: シグナル履歴
**パスパラメータ**: `symbol` (string)
**クエリパラメータ**:
- `days` (int, default=7): 過去日数
- `limit` (int, default=50): 最大シグナル数
- `status` (string, optional): all|executed|expired|cancelled

**レスポンス**: 200 OK

```json
{
  "timestamp": "2024-01-01T12:00:00",
  "symbol": "USDJPY",
  "signals": [
    {
      "signal_id": "sig_001",
      "created_at": "2024-01-01T10:00:00",
      "signal_type": "MARKET_BUY",
      "status": "executed|expired|cancelled",
      "confidence": 0.85,
      "entry_price": 150.25,
      "executed_price": 150.26,
      "outcome": {
        "executed": true,
        "profit_loss": 25.0,
        "accuracy": "correct|incorrect"
      }
    }
  ],
  "statistics": {
    "total_signals": 25,
    "executed_signals": 20,
    "success_rate": 0.75,
    "average_confidence": 0.78,
    "average_profit": 22.5
  }
}
```

#### GET `/api/analysis/health`
**説明**: 分析システムヘルスチェック
**レスポンス**: 200 OK

```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00",
  "components": {
    "dow_theory_analyzer": "healthy",
    "elliott_wave_analyzer": "healthy",
    "unified_analyzer": "healthy",
    "signal_generator": "healthy"
  },
  "performance": {
    "average_analysis_time": "250ms",
    "cache_hit_rate": 0.85,
    "error_rate": 0.02
  }
}
```

---

### 4. MT5制御API (`/api/mt5`)

MetaTrader 5接続管理

#### POST `/api/mt5/connect`
**説明**: MT5サーバーへの接続
**リクエストボディ** (オプション):

```json
{
  "login": 12345,
  "password": "password",
  "server": "XMTrading-MT5 3",
  "timeout": 30000
}
```

**レスポンス**: 200 OK

```json
{
  "success": true,
  "message": "Successfully connected to MT5",
  "connection_status": {
    "connected": true,
    "trade_allowed": true,
    "dlls_allowed": true
  },
  "terminal_info": {
    "version": "5.0.37",
    "build": 3730,
    "name": "MetaTrader 5",
    "path": "/Applications/MetaTrader 5",
    "data_path": "/Users/trader/AppData/Roaming/MetaQuotes/Terminal"
  },
  "account_info": {
    "login": 12345,
    "server": "XMTrading-MT5 3",
    "name": "Demo Account",
    "company": "Tradexfin Limited",
    "currency": "USD",
    "balance": 10000.0,
    "equity": 10000.0,
    "margin": 0.0,
    "free_margin": 10000.0,
    "margin_level": 0.0,
    "leverage": 100,
    "credit": 0.0,
    "profit": 0.0
  },
  "connected_at": "2024-01-01T12:00:00"
}
```

#### POST `/api/mt5/disconnect`
**説明**: MT5からの切断
**レスポンス**: 200 OK

```json
{
  "success": true,
  "message": "Successfully disconnected from MT5",
  "disconnected_at": "2024-01-01T15:00:00",
  "session_duration": "3h 45m"
}
```

#### GET `/api/mt5/status`
**説明**: MT5接続状態確認
**レスポンス**: 200 OK

```json
{
  "connection_status": {
    "connected": true,
    "trade_allowed": true,
    "dlls_allowed": true,
    "tradeapi_disabled": false,
    "trade_expert": true
  },
  "is_connected": true,
  "checked_at": "2024-01-01T12:00:00",
  "terminal_info": {
    "version": "5.0.37",
    "build": 3730,
    "ping_last": "25ms",
    "retransmission": 0.1
  },
  "account_info": {
    "login": 12345,
    "balance": 10000.0,
    "equity": 10000.0,
    "margin": 0.0,
    "free_margin": 10000.0,
    "margin_level": 0.0,
    "profit": 0.0
  },
  "market_status": {
    "market_open": true,
    "symbols_available": 84,
    "last_tick": "2024-01-01T12:00:00"
  }
}
```

#### POST `/api/mt5/reconnect`
**説明**: MT5再接続
**レスポンス**: 200 OK

#### POST `/api/mt5/health-check`
**説明**: MT5ヘルスチェック実行
**レスポンス**: 200 OK

```json
{
  "health_status": "healthy|degraded|unhealthy",
  "timestamp": "2024-01-01T12:00:00",
  "checks": {
    "connection": "pass",
    "trade_allowed": "pass",
    "symbols_available": "pass",
    "account_access": "pass",
    "data_feed": "pass"
  },
  "performance": {
    "ping": "25ms",
    "order_execution_avg": "45ms",
    "data_retrieval_avg": "15ms"
  },
  "recommendations": []
}
```

---

### 5. 設定API (`/api/settings`)

システム構成管理

#### GET `/api/settings/current`
**説明**: 現在のシステム設定取得
**レスポンス**: 200 OK

```json
{
  "timestamp": "2024-01-01T12:00:00",
  "trading_settings": {
    "trading_symbols": ["USDJPY", "EURJPY", "GBPJPY"],
    "max_positions": 3,
    "risk_per_trade": 0.02,
    "enable_trading": true,
    "trading_hours": {
      "start": "09:00",
      "end": "17:00",
      "timezone": "UTC"
    },
    "execution_mode": "demo|live|simulation"
  },
  "risk_management": {
    "max_risk_per_trade": 0.02,
    "max_portfolio_risk": 0.10,
    "max_drawdown_limit": 0.20,
    "emergency_stop_active": false,
    "position_size_method": "fixed_percentage",
    "correlation_limit": 0.7
  },
  "analysis_settings": {
    "dow_weight": 0.4,
    "elliott_weight": 0.6,
    "min_confidence": 0.6,
    "signal_validity_hours": 6,
    "timeframes": ["D1", "H4", "H1"],
    "periods": 100
  },
  "notification_settings": {
    "email_enabled": false,
    "slack_enabled": false,
    "websocket_enabled": true,
    "alert_levels": ["high", "critical"]
  }
}
```

#### PUT `/api/settings/trading`
**説明**: 取引設定更新
**リクエストボディ**:

```json
{
  "trading_symbols": ["USDJPY", "EURJPY"],
  "max_positions": 5,
  "risk_per_trade": 0.015,
  "enable_trading": true,
  "trading_hours": {
    "start": "08:00",
    "end": "18:00"
  }
}
```

**レスポンス**: 200 OK

```json
{
  "success": true,
  "updated_at": "2024-01-01T12:00:00",
  "message": "Trading settings updated successfully",
  "updated_settings": {
    "trading_symbols": ["USDJPY", "EURJPY"],
    "max_positions": 5,
    "risk_per_trade": 0.015
  },
  "restart_required": false
}
```

#### PUT `/api/settings/risk`
**説明**: リスク管理設定更新
**リクエストボディ**:

```json
{
  "max_risk_per_trade": 0.025,
  "max_portfolio_risk": 0.12,
  "max_drawdown_limit": 0.20,
  "position_size_method": "kelly_criterion|volatility_adjusted"
}
```

#### PUT `/api/settings/analysis`
**説明**: 分析設定更新
**リクエストボディ**:

```json
{
  "dow_weight": 0.3,
  "elliott_weight": 0.7,
  "min_confidence": 0.65,
  "signal_validity_hours": 6,
  "timeframes": ["D1", "H4", "H1", "M15"]
}
```

#### GET `/api/settings/system`
**説明**: データベースの全システム設定取得
**レスポンス**: 200 OK

#### PUT `/api/settings/system/{key}`
**説明**: 特定のシステム設定更新
**パスパラメータ**: `key` (string)
**リクエストボディ**:

```json
{
  "key": "setting_name",
  "value": "setting_value",
  "description": "Setting description",
  "type": "string|integer|float|boolean"
}
```

#### DELETE `/api/settings/system/{key}`
**説明**: システム設定削除
**パスパラメータ**: `key` (string)

#### POST `/api/settings/reset`
**説明**: 設定をデフォルトにリセット（安全のため無効化）
**レスポンス**: 403 Forbidden

#### GET `/api/settings/export`
**説明**: 設定のエクスポート
**レスポンス**: 200 OK（JSON設定ファイル）

---

### 6. ダッシュボードUI (`/ui`)

ウェブインターフェースルーティング

#### GET `/ui/dashboard`
**説明**: リアルタイム取引ダッシュボード表示
**レスポンス**: HTML（Glassmorphismデザイン）

#### GET `/ui/`
**説明**: ダッシュボードへのリダイレクト
**レスポンス**: HTMLリダイレクト

---

### 7. WebSocket API (`/ws`)

リアルタイムデータストリーミング

#### 接続: `ws://localhost:8000/ws`

**購読タイプ**:
- `market_data`: リアルタイム価格更新
- `signals`: 取引シグナル通知
- `trading_events`: 注文・ポジションイベント
- `system_status`: システム状態更新
- `positions`: ポジション更新
- `orders`: 注文更新
- `risk_alerts`: リスク管理アラート

**メッセージ形式**:

```json
// 市場データ購読
{
  "type": "subscribe",
  "subscription_type": "market_data",
  "symbols": ["USDJPY", "EURJPY"],
  "client_id": "client_001"
}

// 購読レスポンス
{
  "type": "subscription_response",
  "subscription_type": "market_data",
  "symbols": ["USDJPY", "EURJPY"],
  "success": true,
  "client_id": "client_001",
  "timestamp": "2024-01-01T12:00:00"
}

// 市場データ更新
{
  "type": "market_data",
  "symbol": "USDJPY",
  "data": {
    "bid": 150.25,
    "ask": 150.27,
    "last": 150.26,
    "volume": 1540,
    "change": 0.15,
    "change_percent": 0.10,
    "timestamp": "2024-01-01T12:00:00"
  },
  "timestamp": "2024-01-01T12:00:00"
}

// シグナル通知
{
  "type": "signal",
  "signal_id": "sig_001",
  "symbol": "USDJPY",
  "signal_type": "MARKET_BUY",
  "confidence": 0.85,
  "urgency": "HIGH",
  "timestamp": "2024-01-01T12:00:00"
}

// 取引イベント
{
  "type": "trading_event",
  "event_type": "order_executed|position_opened|position_closed",
  "order_id": "ord_001",
  "symbol": "USDJPY",
  "details": {
    "volume": 0.1,
    "price": 150.25,
    "profit": 25.0
  },
  "timestamp": "2024-01-01T12:00:00"
}

// システム状態
{
  "type": "system_status",
  "status": "healthy|degraded|unhealthy",
  "components": {
    "mt5_connection": "healthy",
    "trading_engine": "active"
  },
  "timestamp": "2024-01-01T12:00:00"
}

// リスクアラート
{
  "type": "risk_alert",
  "alert_level": "low|medium|high|critical",
  "message": "Portfolio risk exceeds 8%",
  "recommendations": ["Consider reducing position sizes"],
  "timestamp": "2024-01-01T12:00:00"
}
```

**利用可能メッセージタイプ**:
- `subscribe/unsubscribe`: 購読管理
- `ping/pong`: 接続ハートビート
- `get_status`: 接続統計取得

---

### 8. コアアプリケーションエンドポイント

#### GET `/`
**説明**: ルートエンドポイント（ダッシュボードへの自動リダイレクト付き）
**レスポンス**: HTML（Glassmorphismデザイン）

#### GET `/health`
**説明**: アプリケーションヘルスチェック
**レスポンス**: 200 OK

```json
{
  "status": "healthy",
  "app_name": "FX Auto Trading System",
  "version": "1.0.0",
  "debug": true,
  "uptime": "2h 45m",
  "timestamp": "2024-01-01T12:00:00"
}
```

#### GET `/info`
**説明**: アプリケーション情報
**レスポンス**: 200 OK

```json
{
  "app_name": "FX Auto Trading System",
  "version": "1.0.0",
  "description": "Automated FX trading system implementing Dow Theory and Elliott Wave analysis",
  "authors": ["Trading System Team"],
  "trading_symbols": ["USDJPY", "EURJPY", "GBPJPY"],
  "supported_timeframes": ["M1", "M5", "M15", "H1", "H4", "D1", "W1", "MN1"],
  "analysis_methods": ["dow_theory", "elliott_wave", "unified"],
  "execution_modes": ["demo", "live", "simulation"],
  "debug_mode": true,
  "last_updated": "2024-01-01T12:00:00"
}
```

---

## エラーハンドリング

**標準エラーレスポンス**:

```json
{
  "detail": "Error description",
  "status_code": 400,
  "error_type": "ValidationError|NotFoundError|InternalError",
  "timestamp": "2024-01-01T12:00:00",
  "request_id": "req_123"
}
```

**一般的なステータスコード**:
- `200`: 成功
- `201`: 作成成功
- `400`: 不正なリクエスト（バリデーションエラー）
- `401`: 認証エラー（未実装）
- `403`: 権限エラー
- `404`: リソースが見つからない
- `422`: バリデーションエラー（詳細）
- `500`: 内部サーバーエラー
- `503`: サービス利用不可（コンポーネント停止）

**詳細エラーレスポンス例**:

```json
{
  "detail": [
    {
      "loc": ["body", "volume"],
      "msg": "ensure this value is greater than 0.01",
      "type": "value_error.number.not_gt",
      "ctx": {"limit_value": 0.01}
    }
  ],
  "status_code": 422,
  "error_type": "ValidationError"
}
```

---

## 認証・セキュリティ

**現在のステータス**: 認証不要（開発モード）
**将来計画**: JWTトークンベース認証

**セキュリティヘッダー**:
- CORS対応
- Content-Type検証
- Request size制限

---

## レート制限・クォータ

**現在のステータス**: レート制限なし
**WebSocket**: コネクションマネージャーによる接続制限
**将来計画**: API Key別レート制限

---

## データ形式

**タイムスタンプ**: ISO 8601形式 (`2024-01-01T12:00:00`)
**価格**: Float（4-5桁精度）
**ボリューム**: Float（FXは0.01最小）
**パーセンテージ**: Float（0.02 = 2%）
**通貨**: USD表示

---

## API文書アクセス

- **インタラクティブ文書**: `http://localhost:8000/docs` (Swagger UI)
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`

---

## パフォーマンス指標

**目標レスポンス時間**:
- 基本エンドポイント: < 50ms
- 分析エンドポイント: < 500ms
- 取引実行: < 100ms
- WebSocket配信: < 10ms

**スループット**:
- REST API: 1000 requests/min
- WebSocket: 10,000 messages/min

---

## 監視・ログ

**ログレベル**:
- DEBUG: 開発情報
- INFO: 一般情報
- WARNING: 注意事項
- ERROR: エラー
- CRITICAL: 緊急事態

**メトリクス**:
- APIレスポンス時間
- エラー率
- 接続数
- 取引実行成功率

---

この包括的なAPIにより、リアルタイムWebSocket統合と広範な監視機能を備えたダッシュボードAPIを通じて、FX取引システムの完全制御が可能です。