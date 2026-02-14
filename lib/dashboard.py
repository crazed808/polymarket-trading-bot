"""
Modern Dashboard for Micro Sniper Bot
Beautiful, colorful terminal UI using Rich
"""

from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich.style import Style
from rich.progress import BarColumn, Progress, TextColumn, SpinnerColumn
from rich import box
from rich.align import Align
from rich.columns import Columns
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import deque
import time


# Modern color palette
COLORS = {
    'primary': '#00D4FF',      # Cyan
    'secondary': '#FF6B6B',    # Coral
    'success': '#00FF88',      # Bright green
    'warning': '#FFD93D',      # Yellow
    'danger': '#FF4757',       # Red
    'info': '#74B9FF',         # Light blue
    'purple': '#A29BFE',       # Purple
    'pink': '#FD79A8',         # Pink
    'orange': '#FDCB6E',       # Orange
    'dark': '#2D3436',         # Dark gray
    'muted': '#636E72',        # Gray
}


@dataclass
class CoinDisplayState:
    """State for displaying a coin in the dashboard."""
    coin: str
    price_up: float = 0.0
    price_down: float = 0.0
    time_remaining: float = 0.0
    status: str = "waiting"
    last_trade_time: float = 0.0
    price_history: List[float] = None

    def __post_init__(self):
        if self.price_history is None:
            self.price_history = []


class ModernDashboard:
    """Modern, colorful dashboard for the micro sniper bot."""

    def __init__(
        self,
        coins: List[str],
        min_price: float = 0.95,
        max_price: float = 0.995,
        min_time: float = 1.0,
        max_time: float = 3.0,
    ):
        self.coins = coins
        self.min_price = min_price
        self.max_price = max_price
        self.min_time = min_time
        self.max_time = max_time

        self.console = Console()
        self.coin_states: Dict[str, CoinDisplayState] = {
            coin: CoinDisplayState(coin=coin) for coin in coins
        }

        # Stats
        self.balance = 0.0
        self.starting_balance = 0.0
        self.total_pnl = 0.0
        self.trades_count = 0
        self.wins = 0
        self.losses = 0
        self.pending_count = 0
        self.filtered_count = 0

        # Events log
        self.events: deque = deque(maxlen=12)
        self.start_time = time.time()

    def update_coin_state(
        self,
        coin: str,
        price_up: float = None,
        price_down: float = None,
        time_remaining: float = None,
        status: str = None,
    ):
        """Update the state of a coin."""
        if coin not in self.coin_states:
            return

        state = self.coin_states[coin]
        if price_up is not None:
            state.price_up = price_up
            state.price_history.append(price_up)
            if len(state.price_history) > 30:
                state.price_history.pop(0)
        if price_down is not None:
            state.price_down = price_down
        if time_remaining is not None:
            state.time_remaining = time_remaining
        if status is not None:
            state.status = status

    def add_event(self, message: str, color: str = "white"):
        """Add an event to the log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.events.append((timestamp, message, color))

    def update_stats(
        self,
        balance: float = None,
        total_pnl: float = None,
        trades_count: int = None,
        wins: int = None,
        losses: int = None,
        pending_count: int = None,
        filtered_count: int = None,
    ):
        """Update dashboard statistics."""
        if balance is not None:
            self.balance = balance
        if total_pnl is not None:
            self.total_pnl = total_pnl
        if trades_count is not None:
            self.trades_count = trades_count
        if wins is not None:
            self.wins = wins
        if losses is not None:
            self.losses = losses
        if pending_count is not None:
            self.pending_count = pending_count
        if filtered_count is not None:
            self.filtered_count = filtered_count

    def _create_sparkline(self, values: List[float], width: int = 10) -> str:
        """Create a mini sparkline from values."""
        if not values:
            return "▁" * width

        # Normalize to 0-1 range
        min_val = min(values)
        max_val = max(values)
        if max_val == min_val:
            return "▄" * min(len(values), width)

        chars = "▁▂▃▄▅▆▇█"
        result = []

        # Take last `width` values
        recent = values[-width:]
        for v in recent:
            normalized = (v - min_val) / (max_val - min_val)
            idx = min(int(normalized * (len(chars) - 1)), len(chars) - 1)
            result.append(chars[idx])

        return "".join(result)

    def _create_header(self) -> Panel:
        """Create the header panel."""
        # Create gradient-style header
        title = Text()
        title.append("⚡ ", style="bold yellow")
        title.append("M", style="bold #FF6B6B")
        title.append("I", style="bold #FF8E6B")
        title.append("C", style="bold #FFB16B")
        title.append("R", style="bold #FFD36B")
        title.append("O", style="bold #FFF66B")
        title.append(" ", style="bold")
        title.append("S", style="bold #D3FF6B")
        title.append("N", style="bold #B1FF6B")
        title.append("I", style="bold #8EFF6B")
        title.append("P", style="bold #6BFF6B")
        title.append("E", style="bold #6BFFB1")
        title.append("R", style="bold #6BFFD3")
        title.append(" ⚡", style="bold yellow")

        # Settings line
        settings = Text()
        settings.append("  Entry: ", style="dim")
        settings.append(f"{self.min_price:.0%}-{self.max_price:.0%}", style=f"bold {COLORS['success']}")
        settings.append("  │  ", style="dim")
        settings.append("Window: ", style="dim")
        settings.append(f"{self.min_time:.0f}s-{self.max_time:.0f}s", style=f"bold {COLORS['info']}")
        settings.append("  │  ", style="dim")
        settings.append("Mode: ", style="dim")
        settings.append("MAX LIQUIDITY", style=f"bold {COLORS['warning']}")

        header_content = Align.center(Group(title, settings))

        return Panel(
            header_content,
            style=f"bold {COLORS['primary']}",
            box=box.DOUBLE_EDGE,
            padding=(0, 2),
        )

    def _create_coin_card(self, coin: str) -> Panel:
        """Create a card for a single coin."""
        state = self.coin_states.get(coin, CoinDisplayState(coin=coin))

        # Determine card border color based on status
        status_colors = {
            "waiting": COLORS['muted'],
            "watching": COLORS['warning'],
            "sniping": COLORS['success'],
            "pending": COLORS['info'],
            "won": COLORS['success'],
            "lost": COLORS['danger'],
        }
        border_color = status_colors.get(state.status, COLORS['muted'])

        # Price display
        content = Text()

        # UP price
        up_prob = state.price_up
        if up_prob >= self.min_price and up_prob <= self.max_price:
            up_style = f"bold {COLORS['success']}"
            up_indicator = "●"
        elif up_prob > self.max_price:
            up_style = f"bold {COLORS['warning']}"
            up_indicator = "◐"
        else:
            up_style = "dim"
            up_indicator = "○"

        content.append(f"{up_indicator} UP   ", style=up_style)
        content.append(f"{up_prob:.1%}\n" if up_prob > 0 else "---\n", style=up_style)

        # DOWN price - show the actual DOWN token price
        # Note: When DOWN token is low (e.g. 0.05), it means DOWN has ~5% probability
        # When DOWN is high (e.g. 0.95), DOWN has ~95% probability
        down_display = state.price_down
        # For highlighting, check if DOWN's probability (1 - price_down for the "other" interpretation) is in range
        # Actually just highlight based on the token price itself
        if down_display >= self.min_price and down_display <= self.max_price:
            down_style = f"bold {COLORS['success']}"
            down_indicator = "●"
        elif down_display > self.max_price:
            down_style = f"bold {COLORS['warning']}"
            down_indicator = "◐"
        elif down_display > 0 and down_display < (1 - self.max_price):
            # DOWN token is very cheap = UP is winning strongly
            down_style = "dim"
            down_indicator = "○"
        else:
            down_style = "dim"
            down_indicator = "○"

        content.append(f"{down_indicator} DOWN ", style=down_style)
        content.append(f"{down_display:.1%}\n" if down_display > 0 else "---\n", style=down_style)

        # Time remaining with visual bar
        content.append("\n")
        if state.time_remaining > 0:
            # Progress towards snipe window
            if state.time_remaining <= self.max_time:
                if state.time_remaining >= self.min_time:
                    time_style = f"bold {COLORS['success']}"
                    bar_char = "█"
                else:
                    time_style = f"bold {COLORS['danger']}"
                    bar_char = "▓"
            else:
                time_style = f"{COLORS['info']}"
                bar_char = "░"

            # Create time bar
            bar_width = 12
            if state.time_remaining <= 60:
                progress = max(0, min(1, (60 - state.time_remaining) / 60))
                filled = int(progress * bar_width)
                bar = bar_char * filled + "░" * (bar_width - filled)
            else:
                bar = "░" * bar_width

            content.append(f"⏱ ", style=time_style)
            content.append(f"{state.time_remaining:>4.0f}s ", style=time_style)
            content.append(bar, style=time_style)
        else:
            content.append("⏱ ", style="dim")
            content.append("---", style="dim")

        # Status badge
        content.append("\n\n")
        status_badges = {
            "waiting": ("  ⏳ WAITING  ", COLORS['muted']),
            "watching": ("  👁  WATCHING ", COLORS['warning']),
            "sniping": ("  🎯 SNIPING! ", COLORS['success']),
            "pending": ("  ⏱  PENDING  ", COLORS['info']),
            "won": ("   ✅ WON!    ", COLORS['success']),
            "lost": ("   ❌ LOST    ", COLORS['danger']),
        }
        badge_text, badge_color = status_badges.get(state.status, ("?", COLORS['muted']))
        content.append(f"{badge_text}\n", style=f"bold {badge_color}")

        # Sparkline
        if state.price_history:
            sparkline = self._create_sparkline(state.price_history, 14)
            content.append(f"  {sparkline}  ", style=f"dim {COLORS['info']}")

        return Panel(
            content,
            title=f"[bold {COLORS['primary']}]{coin}[/]",
            border_style=border_color,
            box=box.ROUNDED,
            padding=(0, 1),
        )

    def _create_stats_panel(self) -> Panel:
        """Create the statistics panel."""
        content = Text()

        # Balance with icon
        content.append("💰 Balance    ", style="white")
        balance_style = COLORS['success'] if self.balance > 0 else COLORS['danger']
        content.append(f"${self.balance:,.2f}\n", style=f"bold {balance_style}")

        # P&L with color
        content.append("📊 P&L        ", style="white")
        pnl_style = COLORS['success'] if self.total_pnl >= 0 else COLORS['danger']
        pnl_sign = "+" if self.total_pnl >= 0 else ""
        content.append(f"{pnl_sign}${self.total_pnl:,.2f}\n", style=f"bold {pnl_style}")

        # Win rate
        total_trades = self.wins + self.losses
        win_rate = (self.wins / total_trades * 100) if total_trades > 0 else 0
        content.append("🎯 Win Rate   ", style="white")
        wr_style = COLORS['success'] if win_rate >= 50 else COLORS['warning'] if win_rate >= 30 else COLORS['danger']
        content.append(f"{win_rate:.0f}%", style=f"bold {wr_style}")
        content.append(f" ({self.wins}W/{self.losses}L)\n", style="dim")

        # Trades & pending
        content.append("📈 Trades     ", style="white")
        content.append(f"{self.trades_count}\n", style=f"bold {COLORS['info']}")

        content.append("⏳ Pending    ", style="white")
        content.append(f"{self.pending_count}\n", style=f"bold {COLORS['warning']}")

        content.append("🚫 Filtered   ", style="white")
        content.append(f"{self.filtered_count}\n", style="dim")

        # Uptime
        uptime = time.time() - self.start_time
        hours, remainder = divmod(int(uptime), 3600)
        minutes, seconds = divmod(remainder, 60)
        content.append("⏱  Uptime     ", style="white")
        content.append(f"{hours:02d}:{minutes:02d}:{seconds:02d}", style="dim")

        return Panel(
            content,
            title=f"[bold {COLORS['purple']}]📊 Stats[/]",
            border_style=COLORS['purple'],
            box=box.ROUNDED,
            padding=(0, 1),
        )

    def _create_events_panel(self) -> Panel:
        """Create the events log panel."""
        content = Text()

        color_map = {
            "green": COLORS['success'],
            "red": COLORS['danger'],
            "yellow": COLORS['warning'],
            "cyan": COLORS['info'],
            "white": "white",
            "dim": "dim",
        }

        if self.events:
            for timestamp, message, color in self.events:
                style = color_map.get(color, "white")
                content.append(f"[dim]{timestamp}[/dim] ")
                content.append(f"{message}\n", style=style)
        else:
            content.append("[dim]Waiting for events...[/dim]")

        return Panel(
            content,
            title=f"[bold {COLORS['orange']}]📋 Events[/]",
            border_style=COLORS['orange'],
            box=box.ROUNDED,
            padding=(0, 1),
        )

    def _create_footer(self, log_file: str = "micro_snipe.log") -> Panel:
        """Create the footer panel."""
        content = Text()
        content.append("📁 ", style="dim")
        content.append(f"{log_file}", style=f"dim italic")
        content.append("  │  ", style="dim")
        content.append("Press ", style="dim")
        content.append("Ctrl+C", style=f"bold {COLORS['warning']}")
        content.append(" to stop", style="dim")
        content.append("  │  ", style="dim")
        content.append(datetime.now().strftime("%H:%M:%S"), style=f"{COLORS['info']}")

        return Panel(
            Align.center(content),
            style="dim",
            box=box.SIMPLE,
            padding=(0, 0),
        )

    def build(self, log_file: str = "micro_snipe.log") -> Layout:
        """Build the complete dashboard layout."""
        layout = Layout()

        # Main structure
        layout.split_column(
            Layout(name="header", size=5),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )

        # Body split
        layout["body"].split_row(
            Layout(name="coins", ratio=3),
            Layout(name="sidebar", ratio=2),
        )

        # Sidebar split
        layout["sidebar"].split_column(
            Layout(name="stats", size=14),
            Layout(name="events"),
        )

        # Create coin cards in a grid
        coin_cards = [self._create_coin_card(coin) for coin in self.coins]

        # Arrange coins in rows of 2
        if len(coin_cards) <= 2:
            coins_layout = Columns(coin_cards, equal=True, expand=True)
        else:
            # Create a grid layout
            rows = []
            for i in range(0, len(coin_cards), 2):
                row = coin_cards[i:i+2]
                rows.append(Columns(row, equal=True, expand=True))
            coins_layout = Group(*rows)

        # Update layouts
        layout["header"].update(self._create_header())
        layout["coins"].update(Panel(
            coins_layout,
            title=f"[bold {COLORS['primary']}]🎯 Markets[/]",
            border_style=COLORS['primary'],
            box=box.ROUNDED,
            padding=(0, 1),
        ))
        layout["stats"].update(self._create_stats_panel())
        layout["events"].update(self._create_events_panel())
        layout["footer"].update(self._create_footer(log_file))

        return layout
