"""
Visualization Module
Creates 52-week charts with trend analysis markers.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pathlib import Path
from typing import Optional
import io


class ChartGenerator:
    """Generates stock charts for Discord display."""
    
    def __init__(self, output_dir: str = "./charts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Style configuration
        plt.style.use('dark_background')
        self.colors = {
            'price': '#00D4AA',
            'sma_20': '#FFD700',
            'sma_50': '#FF6B6B',
            'high': '#00FF00',
            'low': '#FF0000',
            'volume': '#4A90D9',
            'grid': '#333333',
            'background': '#1a1a2e'
        }
    
    def create_52_week_chart(
        self,
        df: pd.DataFrame,
        ticker: str,
        show_signals: bool = True,
        prediction: Optional[dict] = None
    ) -> str:
        """
        Create a comprehensive 52-week chart.
        
        Args:
            df: DataFrame with OHLCV data
            ticker: Stock symbol
            show_signals: Whether to mark 52-week high/low
            prediction: Optional dict with trend prediction info
            
        Returns:
            Path to saved chart image
        """
        fig, axes = plt.subplots(
            2, 1, 
            figsize=(14, 10),
            gridspec_kw={'height_ratios': [3, 1]},
            facecolor=self.colors['background']
        )
        
        ax_price = axes[0]
        ax_volume = axes[1]
        
        # Set background
        for ax in axes:
            ax.set_facecolor(self.colors['background'])
        
        # === Price Chart ===
        dates = df.index
        
        # Main price line
        ax_price.plot(
            dates, df['Close'], 
            color=self.colors['price'], 
            linewidth=1.5, 
            label='Price'
        )
        
        # Moving averages
        if len(df) >= 20:
            sma_20 = df['Close'].rolling(window=20).mean()
            ax_price.plot(
                dates, sma_20, 
                color=self.colors['sma_20'], 
                linewidth=1, 
                alpha=0.7,
                label='20 SMA'
            )
        
        if len(df) >= 50:
            sma_50 = df['Close'].rolling(window=50).mean()
            ax_price.plot(
                dates, sma_50, 
                color=self.colors['sma_50'], 
                linewidth=1, 
                alpha=0.7,
                label='50 SMA'
            )
        
        # 52-week high/low markers
        if show_signals:
            high_52w = df['High'].max()
            low_52w = df['Low'].min()
            high_date = df['High'].idxmax()
            low_date = df['Low'].idxmin()
            
            # High marker
            ax_price.axhline(
                y=high_52w, 
                color=self.colors['high'], 
                linestyle='--', 
                alpha=0.5,
                label=f'52W High: ${high_52w:.2f}'
            )
            ax_price.scatter(
                [high_date], [high_52w], 
                color=self.colors['high'], 
                s=100, 
                marker='^',
                zorder=5
            )
            
            # Low marker
            ax_price.axhline(
                y=low_52w, 
                color=self.colors['low'], 
                linestyle='--', 
                alpha=0.5,
                label=f'52W Low: ${low_52w:.2f}'
            )
            ax_price.scatter(
                [low_date], [low_52w], 
                color=self.colors['low'], 
                s=100, 
                marker='v',
                zorder=5
            )
        
        # Add prediction annotation if provided
        if prediction:
            current_price = df['Close'].iloc[-1]
            pred_text = (
                f"Trend: {prediction.get('direction', 'N/A')}\n"
                f"Predicted: {prediction.get('change', 0):+.1f}%\n"
                f"Signal: {prediction.get('recommendation', 'HOLD')}"
            )
            
            # Color based on recommendation
            rec = prediction.get('recommendation', 'HOLD')
            box_color = {
                'BUY': '#00FF00',
                'SELL': '#FF0000',
                'HOLD': '#FFD700'
            }.get(rec, '#FFD700')
            
            ax_price.annotate(
                pred_text,
                xy=(dates[-1], current_price),
                xytext=(20, 30),
                textcoords='offset points',
                fontsize=10,
                color='white',
                bbox=dict(
                    boxstyle='round,pad=0.5',
                    facecolor=self.colors['background'],
                    edgecolor=box_color,
                    alpha=0.9
                ),
                arrowprops=dict(
                    arrowstyle='->',
                    color=box_color
                )
            )
        
        # Price chart formatting
        ax_price.set_title(
            f'{ticker} - 52 Week Chart',
            fontsize=16,
            fontweight='bold',
            color='white',
            pad=15
        )
        ax_price.set_ylabel('Price ($)', fontsize=12, color='white')
        ax_price.legend(loc='upper left', fontsize=9, facecolor=self.colors['background'])
        ax_price.grid(True, alpha=0.3, color=self.colors['grid'])
        ax_price.tick_params(colors='white')
        
        # === Volume Chart ===
        # Color bars based on price direction
        colors = [
            self.colors['price'] if df['Close'].iloc[i] >= df['Open'].iloc[i] 
            else self.colors['sma_50']
            for i in range(len(df))
        ]
        
        ax_volume.bar(dates, df['Volume'], color=colors, alpha=0.7, width=1)
        
        # Volume moving average
        if len(df) >= 20:
            vol_sma = df['Volume'].rolling(window=20).mean()
            ax_volume.plot(
                dates, vol_sma,
                color=self.colors['sma_20'],
                linewidth=1.5,
                label='20-day avg'
            )
        
        ax_volume.set_ylabel('Volume', fontsize=12, color='white')
        ax_volume.legend(loc='upper left', fontsize=9, facecolor=self.colors['background'])
        ax_volume.grid(True, alpha=0.3, color=self.colors['grid'])
        ax_volume.tick_params(colors='white')
        
        # Format x-axis dates
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add current price annotation
        current_price = df['Close'].iloc[-1]
        ax_price.annotate(
            f'${current_price:.2f}',
            xy=(dates[-1], current_price),
            xytext=(10, 0),
            textcoords='offset points',
            fontsize=11,
            fontweight='bold',
            color=self.colors['price'],
            va='center'
        )
        
        plt.tight_layout()
        
        # Save chart
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{ticker}_{timestamp}.png"
        filepath = self.output_dir / filename
        
        plt.savefig(
            filepath,
            dpi=150,
            facecolor=self.colors['background'],
            edgecolor='none',
            bbox_inches='tight'
        )
        plt.close(fig)
        
        return str(filepath)
    
    def create_chart_bytes(
        self,
        df: pd.DataFrame,
        ticker: str,
        show_signals: bool = True,
        prediction: Optional[dict] = None
    ) -> io.BytesIO:
        """
        Create chart and return as bytes (for Discord attachment).
        
        Returns:
            BytesIO buffer containing the PNG image
        """
        # Create chart (saves to file)
        filepath = self.create_52_week_chart(df, ticker, show_signals, prediction)
        
        # Read back as bytes
        buffer = io.BytesIO()
        with open(filepath, 'rb') as f:
            buffer.write(f.read())
        buffer.seek(0)
        
        return buffer


# Test
if __name__ == "__main__":
    import yfinance as yf
    
    ticker = "AAPL"
    stock = yf.Ticker(ticker)
    df = stock.history(period="1y")
    
    generator = ChartGenerator(output_dir="./test_charts")
    
    prediction = {
        'direction': 'BULLISH',
        'change': 2.3,
        'recommendation': 'BUY'
    }
    
    path = generator.create_52_week_chart(df, ticker, prediction=prediction)
    print(f"Chart saved to: {path}")
