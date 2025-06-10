"""
Analysis Results API endpoints
Phase 7.1 Implementation - Strategy analysis and signal data access
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import asyncio

from app.dependencies import get_settings_cached
from app.config import Settings
from app.analysis.strategy_engine.unified_analyzer import unified_analyzer
from app.analysis.strategy_engine.multi_timeframe_analyzer import multi_timeframe_analyzer
from app.analysis.strategy_engine.signal_generator import signal_generator
from app.analysis.strategy_engine.confidence_calculator import confidence_calculator
from app.analysis.dow_theory.analyzer import dow_theory_analyzer
from app.analysis.elliott_wave.analyzer import elliott_wave_analyzer
from app.mt5.data_fetcher import get_data_fetcher
from app.utils.logger import analysis_logger

router = APIRouter()


# Request/Response Models
class AnalysisRequest(BaseModel):
    symbol: str
    timeframe: str = "H1"
    periods: int = 100

class MultiTimeframeAnalysisRequest(BaseModel):
    symbol: str
    timeframes: List[str] = ["D1", "H4", "H1", "M15"]
    periods: int = 100


# Dow Theory Analysis
@router.get("/dow-theory/{symbol}")
async def get_dow_theory_analysis(
    symbol: str,
    timeframe: str = Query("H1", description="Timeframe for analysis"),
    periods: int = Query(100, description="Number of periods to analyze")
) -> Dict[str, Any]:
    """
    Get Dow Theory analysis for a symbol
    
    Args:
        symbol: Trading symbol
        timeframe: Analysis timeframe
        periods: Number of periods
        
    Returns:
        Dow Theory analysis results
    """
    try:
        data_fetcher = get_data_fetcher()
        
        # Get price data
        # Calculate date range from periods
        end_date = datetime.now()
        # Approximate periods to days (conservative estimate)
        if timeframe == "M1":
            days_back = periods // (24 * 60) + 1
        elif timeframe == "M5":
            days_back = periods // (24 * 12) + 1
        elif timeframe == "M15":
            days_back = periods // (24 * 4) + 1
        elif timeframe == "H1":
            days_back = periods // 24 + 1
        elif timeframe == "H4":
            days_back = periods // 6 + 1
        elif timeframe == "D1":
            days_back = periods + 1
        else:
            days_back = periods + 1  # Default fallback
        start_date = end_date - timedelta(days=days_back)
        
        price_data = await data_fetcher.get_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        if price_data is None or price_data.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No price data available for {symbol} {timeframe}"
            )
        
        # Perform Dow Theory analysis
        analysis_result = dow_theory_analyzer.analyze(price_data, symbol, timeframe)
        
        # Format result
        result = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "timeframe": timeframe,
            "periods_analyzed": len(price_data),
            "analysis": {
                "primary_trend": analysis_result.primary_trend.value if analysis_result.primary_trend else None,
                "secondary_trend": analysis_result.secondary_trend.value if analysis_result.secondary_trend else None,
                "minor_trend": analysis_result.minor_trend.value if analysis_result.minor_trend else None,
                "trend_strength": analysis_result.trend_strength,
                "confirmation_score": analysis_result.confirmation_score,
                "volume_confirmation": analysis_result.volume_confirmation,
                "swing_points": [
                    {
                        "index": sp.index,
                        "price": sp.price,
                        "type": sp.type.value,
                        "timestamp": sp.timestamp.isoformat() if sp.timestamp else None
                    }
                    for sp in analysis_result.swing_points
                ],
                "trend_lines": [
                    {
                        "type": tl.type.value,
                        "start_point": {"index": tl.start_point.index, "price": tl.start_point.price},
                        "end_point": {"index": tl.end_point.index, "price": tl.end_point.price},
                        "strength": tl.strength
                    }
                    for tl in analysis_result.trend_lines
                ]
            },
            "metadata": {
                "analysis_time": analysis_result.analysis_time.isoformat(),
                "data_quality": analysis_result.data_quality,
                "reliability_score": analysis_result.reliability_score
            }
        }
        
        analysis_logger.info(f"Dow Theory analysis completed for {symbol} {timeframe}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        analysis_logger.error(f"Error in Dow Theory analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# Elliott Wave Analysis
@router.get("/elliott-wave/{symbol}")
async def get_elliott_wave_analysis(
    symbol: str,
    timeframe: str = Query("H1", description="Timeframe for analysis"),
    periods: int = Query(100, description="Number of periods to analyze")
) -> Dict[str, Any]:
    """
    Get Elliott Wave analysis for a symbol
    
    Args:
        symbol: Trading symbol
        timeframe: Analysis timeframe
        periods: Number of periods
        
    Returns:
        Elliott Wave analysis results
    """
    try:
        data_fetcher = get_data_fetcher()
        
        # Get price data
        # Calculate date range from periods
        end_date = datetime.now()
        # Approximate periods to days (conservative estimate)
        if timeframe == "M1":
            days_back = periods // (24 * 60) + 1
        elif timeframe == "M5":
            days_back = periods // (24 * 12) + 1
        elif timeframe == "M15":
            days_back = periods // (24 * 4) + 1
        elif timeframe == "H1":
            days_back = periods // 24 + 1
        elif timeframe == "H4":
            days_back = periods // 6 + 1
        elif timeframe == "D1":
            days_back = periods + 1
        else:
            days_back = periods + 1  # Default fallback
        start_date = end_date - timedelta(days=days_back)
        
        price_data = await data_fetcher.get_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        if price_data is None or price_data.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No price data available for {symbol} {timeframe}"
            )
        
        # Perform Elliott Wave analysis
        analysis_result = elliott_wave_analyzer.analyze(price_data, symbol, timeframe)
        
        # Format patterns
        patterns_data = []
        for pattern in analysis_result.patterns:
            pattern_data = {
                "pattern_id": pattern.pattern_id,
                "pattern_type": pattern.pattern_type.value,
                "degree": pattern.degree.value,
                "confidence": pattern.confidence,
                "waves": [
                    {
                        "wave_id": wave.wave_id,
                        "label": wave.label,
                        "start_index": wave.start_index,
                        "end_index": wave.end_index,
                        "start_price": wave.start_price,
                        "end_price": wave.end_price,
                        "price_change": wave.price_change,
                        "time_duration": wave.time_duration
                    }
                    for wave in pattern.waves
                ],
                "fibonacci_levels": pattern.fibonacci_levels,
                "is_complete": pattern.is_complete,
                "validation_score": pattern.validation_score
            }
            patterns_data.append(pattern_data)
        
        # Format predictions
        predictions_data = []
        for prediction in analysis_result.predictions:
            prediction_data = {
                "pattern_id": prediction.pattern_id,
                "next_wave_label": prediction.next_wave_label,
                "scenarios": [
                    {
                        "scenario_id": scenario.scenario_id,
                        "type": scenario.type,
                        "probability": scenario.probability,
                        "price_targets": scenario.price_targets,
                        "time_targets": scenario.time_targets,
                        "risk_levels": scenario.risk_levels
                    }
                    for scenario in prediction.scenarios
                ],
                "confidence": prediction.confidence,
                "fibonacci_targets": prediction.fibonacci_targets
            }
            predictions_data.append(prediction_data)
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "timeframe": timeframe,
            "periods_analyzed": len(price_data),
            "analysis": {
                "patterns": patterns_data,
                "predictions": predictions_data,
                "swing_points": [
                    {
                        "index": sp.index,
                        "price": sp.price,
                        "type": sp.type.value,
                        "timestamp": sp.timestamp.isoformat() if sp.timestamp else None
                    }
                    for sp in analysis_result.swing_points
                ],
                "fibonacci_levels": analysis_result.fibonacci_levels
            },
            "summary": {
                "total_patterns": len(analysis_result.patterns),
                "complete_patterns": len([p for p in analysis_result.patterns if p.is_complete]),
                "predictions_count": len(analysis_result.predictions),
                "average_confidence": sum(p.confidence for p in analysis_result.patterns) / max(len(analysis_result.patterns), 1)
            },
            "metadata": {
                "analysis_time": analysis_result.analysis_time.isoformat(),
                "reliability_score": analysis_result.reliability_score
            }
        }
        
        analysis_logger.info(f"Elliott Wave analysis completed for {symbol} {timeframe}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        analysis_logger.error(f"Error in Elliott Wave analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# Unified Analysis
@router.get("/unified/{symbol}")
async def get_unified_analysis(
    symbol: str,
    timeframe: str = Query("H1", description="Timeframe for analysis"),
    periods: int = Query(100, description="Number of periods to analyze")
) -> Dict[str, Any]:
    """
    Get unified analysis combining Dow Theory and Elliott Wave
    
    Args:
        symbol: Trading symbol
        timeframe: Analysis timeframe
        periods: Number of periods
        
    Returns:
        Unified analysis results
    """
    try:
        data_fetcher = get_data_fetcher()
        
        # Get price data
        # Calculate date range from periods
        end_date = datetime.now()
        # Approximate periods to days (conservative estimate)
        if timeframe == "M1":
            days_back = periods // (24 * 60) + 1
        elif timeframe == "M5":
            days_back = periods // (24 * 12) + 1
        elif timeframe == "M15":
            days_back = periods // (24 * 4) + 1
        elif timeframe == "H1":
            days_back = periods // 24 + 1
        elif timeframe == "H4":
            days_back = periods // 6 + 1
        elif timeframe == "D1":
            days_back = periods + 1
        else:
            days_back = periods + 1  # Default fallback
        start_date = end_date - timedelta(days=days_back)
        
        price_data = await data_fetcher.get_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        if price_data is None or price_data.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No price data available for {symbol} {timeframe}"
            )
        
        # Perform unified analysis
        unified_result = unified_analyzer.analyze(price_data, symbol, timeframe)
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "timeframe": timeframe,
            "periods_analyzed": len(price_data),
            "unified_analysis": {
                "combined_signal": unified_result.combined_signal,
                "combined_confidence": unified_result.combined_confidence,
                "agreement_score": unified_result.agreement_score,
                "dow_analysis": {
                    "signal": unified_result.dow_signal,
                    "confidence": unified_result.dow_confidence,
                    "trend": unified_result.dow_trend.primary_trend.value if unified_result.dow_trend else None
                },
                "elliott_analysis": {
                    "signal": unified_result.elliott_signal,
                    "confidence": unified_result.elliott_confidence,
                    "patterns_count": len(unified_result.elliott_patterns)
                },
                "price_targets": unified_result.price_targets,
                "risk_levels": unified_result.risk_levels,
                "swing_points": [
                    {
                        "index": sp.index,
                        "price": sp.price,
                        "type": sp.type.value,
                        "source": sp.source
                    }
                    for sp in unified_result.swing_points
                ]
            },
            "recommendations": {
                "action": unified_result.combined_signal,
                "confidence_grade": "A" if unified_result.combined_confidence > 0.8 else
                                  "B" if unified_result.combined_confidence > 0.6 else
                                  "C" if unified_result.combined_confidence > 0.4 else "D",
                "risk_reward_favorable": unified_result.agreement_score > 0.6,
                "entry_timing": "immediate" if unified_result.combined_confidence > 0.8 else
                               "wait_for_confirmation" if unified_result.combined_confidence > 0.5 else "avoid"
            }
        }
        
        analysis_logger.info(f"Unified analysis completed for {symbol} {timeframe}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        analysis_logger.error(f"Error in unified analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# Multi-Timeframe Analysis
@router.post("/multi-timeframe")
async def get_multi_timeframe_analysis(request: MultiTimeframeAnalysisRequest) -> Dict[str, Any]:
    """
    Get multi-timeframe analysis
    
    Args:
        request: Multi-timeframe analysis parameters
        
    Returns:
        Multi-timeframe analysis results
    """
    try:
        data_fetcher = get_data_fetcher()
        
        # Get price data for all timeframes
        timeframe_data = {}
        for tf in request.timeframes:
            try:
                # Calculate date range from periods
                end_date = datetime.now()
                # Approximate periods to days (conservative estimate)
                if tf == "M1":
                    days_back = request.periods // (24 * 60) + 1
                elif tf == "M5":
                    days_back = request.periods // (24 * 12) + 1
                elif tf == "M15":
                    days_back = request.periods // (24 * 4) + 1
                elif tf == "H1":
                    days_back = request.periods // 24 + 1
                elif tf == "H4":
                    days_back = request.periods // 6 + 1
                elif tf == "D1":
                    days_back = request.periods + 1
                else:
                    days_back = request.periods + 1  # Default fallback
                start_date = end_date - timedelta(days=days_back)
                
                price_data = await data_fetcher.get_historical_data(
                    symbol=request.symbol,
                    timeframe=tf,
                    start_date=start_date,
                    end_date=end_date
                )
                if price_data is not None and not price_data.empty:
                    timeframe_data[tf] = price_data
            except Exception as e:
                analysis_logger.warning(f"Failed to get data for {request.symbol} {tf}: {e}")
        
        if not timeframe_data:
            raise HTTPException(
                status_code=404,
                detail=f"No price data available for {request.symbol}"
            )
        
        # Perform multi-timeframe analysis
        mtf_result = multi_timeframe_analyzer.analyze_multiple_timeframes(
            timeframe_data, request.symbol
        )
        
        # Format timeframe results
        timeframe_results = {}
        for tf, result in mtf_result.timeframe_results.items():
            timeframe_results[tf] = {
                "signal": result.combined_signal,
                "confidence": result.combined_confidence,
                "agreement_score": result.agreement_score,
                "dow_confidence": result.dow_confidence,
                "elliott_confidence": result.elliott_confidence
            }
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "symbol": request.symbol,
            "timeframes_analyzed": list(timeframe_data.keys()),
            "multi_timeframe_analysis": {
                "primary_signal": mtf_result.primary_signal,
                "primary_confidence": mtf_result.primary_confidence,
                "alignment_score": mtf_result.alignment_score,
                "trend_consistency": mtf_result.trend_consistency,
                "optimal_entry_timeframe": mtf_result.optimal_entry_timeframe,
                "entry_confirmation_score": mtf_result.entry_confirmation_score,
                "conflicting_timeframes": mtf_result.conflicting_timeframes,
                "timeframe_hierarchy": mtf_result.timeframe_hierarchy,
                "timeframe_results": timeframe_results
            },
            "recommendations": {
                "overall_action": mtf_result.primary_signal,
                "confidence_level": "high" if mtf_result.primary_confidence > 0.7 else
                                  "medium" if mtf_result.primary_confidence > 0.5 else "low",
                "timeframe_alignment": "strong" if mtf_result.alignment_score > 0.7 else
                                     "moderate" if mtf_result.alignment_score > 0.5 else "weak",
                "entry_timeframe": mtf_result.optimal_entry_timeframe,
                "wait_for_alignment": mtf_result.alignment_score < 0.6
            }
        }
        
        analysis_logger.info(f"Multi-timeframe analysis completed for {request.symbol}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        analysis_logger.error(f"Error in multi-timeframe analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# Signal Generation
@router.get("/signals/{symbol}")
async def generate_signal(
    symbol: str,
    timeframe: str = Query("H1", description="Timeframe for analysis"),
    periods: int = Query(100, description="Number of periods to analyze")
) -> Dict[str, Any]:
    """
    Generate trading signal from analysis
    
    Args:
        symbol: Trading symbol
        timeframe: Analysis timeframe
        periods: Number of periods
        
    Returns:
        Generated trading signal
    """
    try:
        data_fetcher = get_data_fetcher()
        
        # Get price data
        # Calculate date range from periods
        end_date = datetime.now()
        # Approximate periods to days (conservative estimate)
        if timeframe == "M1":
            days_back = periods // (24 * 60) + 1
        elif timeframe == "M5":
            days_back = periods // (24 * 12) + 1
        elif timeframe == "M15":
            days_back = periods // (24 * 4) + 1
        elif timeframe == "H1":
            days_back = periods // 24 + 1
        elif timeframe == "H4":
            days_back = periods // 6 + 1
        elif timeframe == "D1":
            days_back = periods + 1
        else:
            days_back = periods + 1  # Default fallback
        start_date = end_date - timedelta(days=days_back)
        
        price_data = await data_fetcher.get_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        if price_data is None or price_data.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No price data available for {symbol} {timeframe}"
            )
        
        # Get current price
        current_price = price_data[-1]['close']
        
        # Perform unified analysis
        unified_result = unified_analyzer.analyze(price_data, symbol, timeframe)
        
        # Generate signal
        signal = signal_generator.generate_signal_from_unified(unified_result, current_price)
        
        if not signal:
            return {
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "timeframe": timeframe,
                "signal_generated": False,
                "reason": "Analysis did not meet signal generation criteria",
                "analysis_summary": {
                    "combined_confidence": unified_result.combined_confidence,
                    "agreement_score": unified_result.agreement_score,
                    "combined_signal": unified_result.combined_signal
                }
            }
        
        # Calculate confidence
        confidence_result = confidence_calculator.calculate_unified_confidence(unified_result, signal)
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "timeframe": timeframe,
            "signal_generated": True,
            "signal": {
                "signal_id": signal.signal_id,
                "signal_type": signal.signal_type.value,
                "urgency": signal.urgency.value,
                "entry_price": signal.entry_price,
                "stop_loss": signal.stop_loss,
                "take_profit_1": signal.take_profit_1,
                "take_profit_2": signal.take_profit_2,
                "take_profit_3": signal.take_profit_3,
                "confidence": signal.confidence,
                "strength": signal.strength,
                "quality_score": signal.quality_score,
                "risk_reward_ratio": signal.risk_reward_ratio,
                "position_size_pct": signal.position_size_pct,
                "valid_until": signal.valid_until.isoformat() if signal.valid_until else None
            },
            "confidence_analysis": {
                "overall_confidence": confidence_result.overall_confidence,
                "confidence_grade": confidence_result.confidence_grade,
                "confidence_factors": confidence_result.confidence_factors,
                "risk_assessment": confidence_result.risk_assessment
            },
            "analysis_summary": signal.analysis_summary
        }
        
        analysis_logger.info(f"Signal generated for {symbol} {timeframe}: {signal.signal_type.value}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        analysis_logger.error(f"Error generating signal: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# Historical Signals
@router.get("/signals/{symbol}/history")
async def get_signal_history(
    symbol: str,
    days: int = Query(7, description="Number of days to look back"),
    limit: int = Query(50, description="Maximum number of signals to return")
) -> Dict[str, Any]:
    """
    Get historical signals for a symbol
    
    Args:
        symbol: Trading symbol
        days: Number of days to look back
        limit: Maximum signals to return
        
    Returns:
        Historical signals
    """
    try:
        # This would typically query the database for stored signals
        # For now, return mock historical data structure
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "query_period_days": days,
            "signals": [],  # Would be populated from database
            "summary": {
                "total_signals": 0,
                "buy_signals": 0,
                "sell_signals": 0,
                "avg_confidence": 0.0,
                "signal_frequency_per_day": 0.0
            },
            "note": "Historical signal storage will be implemented in database integration phase"
        }
        
        analysis_logger.info(f"Signal history requested for {symbol}")
        return result
        
    except Exception as e:
        analysis_logger.error(f"Error getting signal history: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )