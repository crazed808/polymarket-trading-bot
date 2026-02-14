"""
Trading Bot Module - Main Trading Interface

A production-ready trading bot for Polymarket with:
- Gasless transactions via Builder Program
- Encrypted private key storage
- Modular strategy support
- Comprehensive order management

Example:
    from src.bot import TradingBot

    # Initialize with config
    bot = TradingBot(config_path="config.yaml")

    # Or manually
    bot = TradingBot(
        safe_address="0x...",
        builder_creds=builder_creds,
        private_key="0x..."  # or use encrypted key
    )

    # Place an order
    result = await bot.place_order(
        token_id="123...",
        price=0.65,
        size=10,
        side="BUY"
    )
"""

import os
import asyncio
import logging
from typing import Optional, Dict, Any, List, Callable, TypeVar
from dataclasses import dataclass, field
from enum import Enum

from .config import Config, BuilderConfig
from .signer import OrderSigner, Order
from .client import ApiCredentials
from py_clob_client.client import ClobClient as OfficialClobClient
from py_clob_client.clob_types import OrderArgs, OrderType, MarketOrderArgs, BalanceAllowanceParams, AssetType
from .crypto import KeyManager, CryptoError, InvalidPasswordError

# Official Polymarket relayer client
from py_builder_relayer_client.client import RelayClient
from py_builder_relayer_client.models import SafeTransaction, OperationType, RelayerTransactionState
from py_builder_signing_sdk.config import BuilderConfig as OfficialBuilderConfig
from py_builder_signing_sdk.sdk_types import BuilderApiKeyCreds


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

T = TypeVar("T")

class OrderSide(str, Enum):
    """Order side constants."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order type constants."""
    GTC = "GTC"  # Good Till Cancelled
    GTD = "GTD"  # Good Till Date
    FOK = "FOK"  # Fill Or Kill


@dataclass
class OrderResult:
    """Result of an order operation."""
    success: bool
    order_id: Optional[str] = None
    status: Optional[str] = None
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_response(cls, response: Dict[str, Any]) -> "OrderResult":
        """Create from API response."""
        success = response.get("success", False)
        error_msg = response.get("errorMsg", "")

        return cls(
            success=success,
            order_id=response.get("orderId"),
            status=response.get("status"),
            message=error_msg if not success else "Order placed successfully",
            data=response
        )


class TradingBotError(Exception):
    """Base exception for trading bot errors."""
    pass


class NotInitializedError(TradingBotError):
    """Raised when bot is not initialized."""
    pass


class TradingBot:
    """
    Main trading bot class for Polymarket.

    Provides a high-level interface for:
    - Order placement and cancellation
    - Position management
    - Trade history
    - Gasless transactions (with Builder Program)

    Attributes:
        config: Bot configuration
        signer: Order signer instance
        clob_client: CLOB API client
        relayer_client: Relayer API client (if gasless enabled)
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[Config] = None,
        safe_address: Optional[str] = None,
        builder_creds: Optional[BuilderConfig] = None,
        private_key: Optional[str] = None,
        encrypted_key_path: Optional[str] = None,
        password: Optional[str] = None,
        api_creds_path: Optional[str] = None,
        log_level: int = logging.INFO
    ):
        """
        Initialize trading bot.

        Can be initialized in multiple ways:

        1. From config file:
           bot = TradingBot(config_path="config.yaml")

        2. From Config object:
           bot = TradingBot(config=my_config)

        3. With manual parameters:
           bot = TradingBot(
               safe_address="0x...",
               builder_creds=builder_creds,
               private_key="0x..."
           )

        4. With encrypted key:
           bot = TradingBot(
               safe_address="0x...",
               encrypted_key_path="credentials/key.enc",
               password="mypassword"
           )

        Args:
            config_path: Path to config YAML file
            config: Config object
            safe_address: Safe/Proxy wallet address
            builder_creds: Builder Program credentials
            private_key: Raw private key (with 0x prefix)
            encrypted_key_path: Path to encrypted key file
            password: Password for encrypted key
            api_creds_path: Path to API credentials file
            log_level: Logging level
        """
        # Set log level
        logger.setLevel(log_level)

        # Load configuration (merge YAML with environment variables)
        if config_path:
            self.config = Config.load_with_env(config_path)
        elif config:
            self.config = config
        else:
            self.config = Config.from_env()

        # Override with provided parameters
        if safe_address:
            self.config.safe_address = safe_address
        if builder_creds:
            self.config.builder = builder_creds
            self.config.use_gasless = True

        # Initialize components
        self.signer: Optional[OrderSigner] = None
        self.clob_client: Optional[ClobClient] = None
        self.relayer_client: Optional[RelayClient] = None
        self._api_creds: Optional[ApiCredentials] = None

        # Load private key
        if private_key:
            self.signer = OrderSigner(private_key)
        elif encrypted_key_path and password:
            self._load_encrypted_key(encrypted_key_path, password)

        # Load API credentials
        if api_creds_path:
            self._load_api_creds(api_creds_path)

        # Initialize API clients
        self._init_clients()

        # Auto-derive API credentials if we have a signer but no API creds
        if self.signer and not self._api_creds:
            self._derive_api_creds()

        logger.info(f"TradingBot initialized (gasless: {self.config.use_gasless})")

    def _load_encrypted_key(self, filepath: str, password: str) -> None:
        """Load and decrypt private key from encrypted file."""
        try:
            manager = KeyManager()
            private_key = manager.load_and_decrypt(password, filepath)
            self.signer = OrderSigner(private_key)
            logger.info(f"Loaded encrypted key from {filepath}")
        except FileNotFoundError:
            raise TradingBotError(f"Encrypted key file not found: {filepath}")
        except InvalidPasswordError:
            raise TradingBotError("Invalid password for encrypted key")
        except CryptoError as e:
            raise TradingBotError(f"Failed to load encrypted key: {e}")

    def _load_api_creds(self, filepath: str) -> None:
        """Load API credentials from file."""
        if os.path.exists(filepath):
            try:
                self._api_creds = ApiCredentials.load(filepath)
                logger.info(f"Loaded API credentials from {filepath}")
            except Exception as e:
                logger.warning(f"Failed to load API credentials: {e}")

    def _derive_api_creds(self) -> None:
        """Derive L2 API credentials from signer."""
        if not self.signer or not self.clob_client:
            return

        # API credentials are now handled in _init_clients
        pass

    def _init_clients(self) -> None:
        """Initialize API clients."""
        # CLOB client
        # Get private key from signer
        pk = None
        if self.signer and hasattr(self.signer, 'wallet'):
            pk = self.signer.wallet.key.hex()
        
        if not pk:
            raise ValueError("Private key required for ClobClient")
        
        self.clob_client = OfficialClobClient(
            host=self.config.clob.host,
            key=pk,
            chain_id=self.config.clob.chain_id,
            signature_type=self.config.clob.signature_type,
            funder=self.config.safe_address,
        )
        
        # Derive and set API credentials
        creds = self.clob_client.create_or_derive_api_creds()
        self.clob_client.set_api_creds(creds)
        logger.info("API credentials set successfully")

        # Relayer client (for gasless)
        if self.config.use_gasless:
            # Convert our BuilderConfig to the official SDK's BuilderConfig
            builder_creds = BuilderApiKeyCreds(
                key=self.config.builder.api_key,
                secret=self.config.builder.api_secret,
                passphrase=self.config.builder.api_passphrase
            )
            official_builder_config = OfficialBuilderConfig(
                local_builder_creds=builder_creds
            )

            self.relayer_client = RelayClient(
                relayer_url=self.config.relayer.host,
                chain_id=self.config.clob.chain_id,
                private_key=pk,
                builder_config=official_builder_config
            )
            logger.info("Relayer client initialized (gasless enabled)")

    async def _run_in_thread(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Run a blocking call in a worker thread to avoid event loop stalls."""
        return await asyncio.to_thread(func, *args, **kwargs)

    def is_initialized(self) -> bool:
        """Check if bot is properly initialized."""
        return (
            self.signer is not None and
            self.config.safe_address and
            self.clob_client is not None
        )

    def require_signer(self) -> OrderSigner:
        """Get signer or raise if not initialized."""
        if not self.signer:
            raise NotInitializedError(
                "Signer not initialized. Provide private_key or encrypted_key."
            )
        return self.signer

    async def place_order(
        self,
        token_id: str,
        price: float,
        size: float,
        side: str,
        order_type: str = "GTC",
        fee_rate_bps: int = 0
    ) -> OrderResult:
        """
        Place a limit order using official SDK.
        
        Args:
            token_id: Market token ID
            price: Price per share (0-1)
            size: Size in shares or USD depending on SDK
            side: 'BUY' or 'SELL'
            order_type: Order type (GTC, GTD, FOK)
            fee_rate_bps: Fee rate in basis points (ignored, SDK handles this)
            
        Returns:
            OrderResult with order status
        """
        try:
            # Create order args for official SDK
            logger.info(f"[ORDER] Creating order: {side} {size} @ {price} on token {token_id[:20]}...")
            order_args = OrderArgs(
                price=price,
                size=size,
                side=side.upper(),
                token_id=token_id
            )
            
            # Place order using official SDK (creates AND posts to exchange)
            logger.info(f"[ORDER] Calling create_and_post_order...")
            response = await self._run_in_thread(
                self.clob_client.create_and_post_order,
                order_args
            )
            
            logger.info(f"[ORDER] Response type: {type(response)}, response: {response}")
            logger.info(f"Order placed: {side} {size} @ {price}")
            
            # Convert SignedOrder to dict if needed
            if hasattr(response, 'dict'):
                response_dict = response.dict()
            elif isinstance(response, dict):
                response_dict = response
            else:
                response_dict = {"response": str(response)}
            
            order_id = response_dict.get("orderID", "") or response_dict.get("order_id", "")
            
            return OrderResult(
                success=True,
                order_id=str(order_id),
                message=f"Order placed successfully",
                data=response_dict
            )
            
        except Exception as e:
            logger.error(f"Failed to place order: {e}", exc_info=True)
            logger.error(f"[ORDER] Error details - token_id: {token_id}, price: {price}, size: {size}, side: {side}")
            return OrderResult(
                success=False,
                message=f"Order failed: {str(e)}"
            )

    async def pre_warm_token(self, token_id: str):
        """Pre-cache tick_size, neg_risk, and fee_rate for a token to avoid
        API lookups during time-critical order creation."""
        try:
            await self._run_in_thread(self.clob_client.get_tick_size, token_id)
            await self._run_in_thread(self.clob_client.get_neg_risk, token_id)
            await self._run_in_thread(self.clob_client.get_fee_rate_bps, token_id)
            logger.debug(f"[PRE-WARM] Cached tick_size/neg_risk/fee_rate for {token_id[:20]}...")
        except Exception as e:
            logger.warning(f"[PRE-WARM] Failed for {token_id[:20]}: {e}")

    async def create_signed_order(
        self,
        token_id: str,
        price: float,
        size: float,
        side: str,
    ):
        """Create and sign an order WITHOUT posting it. Returns the signed
        order object that can later be posted with post_signed_order().

        Call pre_warm_token() first to ensure SDK caches are populated."""
        order_args = OrderArgs(
            price=price,
            size=size,
            side=side.upper(),
            token_id=token_id
        )
        signed_order = await self._run_in_thread(
            self.clob_client.create_order,
            order_args
        )
        return signed_order

    async def post_signed_order(
        self,
        signed_order,
        order_type: str = "GTC",
    ) -> OrderResult:
        """Post a previously signed order to the exchange.

        Args:
            signed_order: Order object returned by create_signed_order()
            order_type: "GTC", "FOK", or "FAK" (Fill And Kill / IOC equivalent)
        """
        try:
            response = await self._run_in_thread(
                self.clob_client.post_order,
                signed_order,
                order_type
            )

            if hasattr(response, 'dict'):
                response_dict = response.dict()
            elif isinstance(response, dict):
                response_dict = response
            else:
                response_dict = {"response": str(response)}

            order_id = response_dict.get("orderID", "") or response_dict.get("order_id", "")

            return OrderResult(
                success=True,
                order_id=str(order_id),
                message="Order placed successfully",
                data=response_dict
            )
        except Exception as e:
            logger.error(f"Failed to post signed order: {e}", exc_info=True)
            return OrderResult(
                success=False,
                message=f"Order post failed: {str(e)}"
            )

    async def place_market_order(
        self,
        token_id: str,
        amount: float,
        side: str,
        worst_price: float = 0.0,
    ) -> OrderResult:
        """
        Place a market order using FOK (Fill Or Kill).

        Unlike limit orders, market orders fill immediately at the best available
        price or fail entirely. This ensures we actually receive tokens before
        opening a position.

        Args:
            token_id: Market token ID
            amount: Amount in USD to spend (for BUY) or shares to sell (for SELL)
            side: 'BUY' or 'SELL'
            worst_price: Worst acceptable price (0 = any price)

        Returns:
            OrderResult with order status
        """
        try:
            logger.info(f"[MARKET ORDER] Creating: {side} ${amount} on token {token_id[:20]}...")

            # Create market order args (FOK by default)
            market_args = MarketOrderArgs(
                token_id=token_id,
                amount=amount,
                side=side.upper(),
                price=worst_price,  # 0 = any price
            )

            # Create signed market order
            logger.info(f"[MARKET ORDER] Creating signed order...")
            signed_order = await self._run_in_thread(
                self.clob_client.create_market_order,
                market_args
            )

            # Post the order
            logger.info(f"[MARKET ORDER] Posting order...")
            response = await self._run_in_thread(
                self.clob_client.post_order,
                signed_order,
                "FOK"  # Fill Or Kill
            )

            logger.info(f"[MARKET ORDER] Response type: {type(response).__name__}")
            logger.info(f"[MARKET ORDER] Response: {response}")

            # Parse response
            if isinstance(response, dict):
                response_dict = response
            elif hasattr(response, 'dict'):
                response_dict = response.dict()
            elif hasattr(response, '__dict__'):
                response_dict = response.__dict__
            else:
                response_dict = {"response": str(response)}

            # Log all keys for debugging
            logger.info(f"[MARKET ORDER] Response keys: {list(response_dict.keys())}")

            success = response_dict.get("success", True)
            error_msg = response_dict.get("errorMsg", "") or response_dict.get("error_msg", "")
            status = response_dict.get("status", "")
            taking_amount = response_dict.get("takingAmount", "0") or response_dict.get("taking_amount", "0")
            making_amount = response_dict.get("makingAmount", "0") or response_dict.get("making_amount", "0")

            logger.info(f"[MARKET ORDER] Parsed: success={success}, status={status}, taking={taking_amount}, making={making_amount}, error={error_msg}")

            # Check for FOK not filled error specifically
            if error_msg and "FOK" in error_msg.upper() and "NOT" in error_msg.upper() and "FILLED" in error_msg.upper():
                logger.warning(f"FOK order not filled: {error_msg}")
                return OrderResult(
                    success=False,
                    message=f"FOK order not filled",
                    data=response_dict
                )

            # Check for other errors in errorMsg
            if error_msg and any(x in error_msg.upper() for x in ["ERROR", "INVALID", "NOT_ENOUGH"]):
                logger.warning(f"Market order failed: {error_msg}")
                return OrderResult(
                    success=False,
                    message=f"Market order failed: {error_msg}",
                    data=response_dict
                )

            # Check for explicit server-side failure
            if success is False:
                logger.warning(f"Market order server error: {error_msg}")
                return OrderResult(
                    success=False,
                    message=f"Server error: {error_msg}",
                    data=response_dict
                )

            # For FOK orders, check status - "matched" means filled
            # Other statuses like "delayed" or empty might still be valid
            if status and status.lower() == "matched":
                logger.info(f"FOK order matched (filled): status={status}")
            elif status:
                logger.info(f"Order status: {status}")

            # Parse amounts for logging (but don't use for success/fail determination)
            try:
                taking_value = float(taking_amount) if taking_amount else 0
                making_value = float(making_amount) if making_amount else 0
            except (ValueError, TypeError):
                taking_value = 0
                making_value = 0

            order_id = response_dict.get("orderID", "") or response_dict.get("order_id", "")

            logger.info(f"Market order filled: taking={taking_amount}, making={making_amount}")

            return OrderResult(
                success=True,
                order_id=str(order_id),
                message="Market order filled",
                data=response_dict
            )

        except Exception as e:
            logger.error(f"Failed to place market order: {e}", exc_info=True)
            return OrderResult(
                success=False,
                message=f"Market order failed: {str(e)}"
            )

    async def get_balance_allowance(
        self,
        token_id: str,
        asset_type: str = "CONDITIONAL"
    ) -> Dict[str, Any]:
        """
        Get balance and allowance for a token.

        Args:
            token_id: Token ID to check
            asset_type: "COLLATERAL" for USDC, "CONDITIONAL" for outcome tokens

        Returns:
            Dict with 'balance' and 'allowance' as strings
        """
        try:
            # Convert string asset_type to enum
            at = AssetType.COLLATERAL if asset_type == "COLLATERAL" else AssetType.CONDITIONAL
            params = BalanceAllowanceParams(asset_type=at, token_id=token_id)
            result = await self._run_in_thread(
                self.clob_client.get_balance_allowance,
                params
            )
            logger.debug(f"Balance/allowance for {token_id[:20] if token_id else 'USDC'}: {result}")
            return result
        except Exception as e:
            logger.error(f"Failed to get balance/allowance: {e}")
            return {"balance": "0", "allowance": "0"}

    async def has_tokens(self, token_id: str, required_amount: float) -> bool:
        """
        Check if we have enough tokens to sell.

        Args:
            token_id: Token ID to check
            required_amount: Amount needed

        Returns:
            True if balance >= required_amount
        """
        try:
            ba = await self.get_balance_allowance(token_id, "CONDITIONAL")
            balance = float(ba.get("balance", "0"))
            has_enough = balance >= required_amount
            logger.info(f"Token balance check: have {balance}, need {required_amount}, ok={has_enough}")
            return has_enough
        except Exception as e:
            logger.error(f"Failed to check token balance: {e}")
            return False

    async def place_orders(
        self,
        orders: List[Dict[str, Any]],
        order_type: str = "GTC"
    ) -> List[OrderResult]:
        """
        Place multiple orders.

        Args:
            orders: List of order dictionaries with keys:
                - token_id: Market token ID
                - price: Price per share
                - size: Number of shares
                - side: 'BUY' or 'SELL'
            order_type: Order type (GTC, GTD, FOK)

        Returns:
            List of OrderResults
        """
        results = []
        for order_data in orders:
            result = await self.place_order(
                token_id=order_data["token_id"],
                price=order_data["price"],
                size=order_data["size"],
                side=order_data["side"],
                order_type=order_type,
            )
            results.append(result)

            # Small delay between orders to avoid rate limits
            await asyncio.sleep(0.1)

        return results

    async def cancel_order(self, order_id: str) -> OrderResult:
        """
        Cancel a specific order.

        Args:
            order_id: Order ID to cancel

        Returns:
            OrderResult with cancellation status
        """
        try:
            response = await self._run_in_thread(self.clob_client.cancel_order, order_id)
            logger.info(f"Order cancelled: {order_id}")
            return OrderResult(
                success=True,
                order_id=order_id,
                message="Order cancelled",
                data=response
            )
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return OrderResult(
                success=False,
                order_id=order_id,
                message=str(e)
            )

    async def cancel_all_orders(self) -> OrderResult:
        """
        Cancel all open orders.

        Returns:
            OrderResult with cancellation status
        """
        try:
            response = await self._run_in_thread(self.clob_client.cancel_all)
            logger.info("All orders cancelled")
            return OrderResult(
                success=True,
                message="All orders cancelled",
                data=response
            )
        except Exception as e:
            logger.error(f"Failed to cancel orders: {e}")
            return OrderResult(success=False, message=str(e))

    async def cancel_market_orders(
        self,
        market: Optional[str] = None,
        asset_id: Optional[str] = None
    ) -> OrderResult:
        """
        Cancel orders for a specific market.

        Args:
            market: Condition ID of the market (optional)
            asset_id: Token/asset ID (optional)

        Returns:
            OrderResult with cancellation status
        """
        try:
            response = await self._run_in_thread(
                self.clob_client.cancel_market_orders,
                market,
                asset_id,
            )
            logger.info(f"Market orders cancelled (market: {market or 'all'}, asset: {asset_id or 'all'})")
            return OrderResult(
                success=True,
                message=f"Orders cancelled for market {market or 'all'}",
                data=response
            )
        except Exception as e:
            logger.error(f"Failed to cancel market orders: {e}")
            return OrderResult(success=False, message=str(e))

    async def get_open_orders(self) -> List[Dict[str, Any]]:
        """
        Get all open orders.

        Returns:
            List of open orders
        """
        try:
            orders = await self._run_in_thread(self.clob_client.get_orders)
            logger.debug(f"Retrieved {len(orders)} open orders")
            return orders
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []

    async def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Get order details.

        Args:
            order_id: Order ID

        Returns:
            Order details or None
        """
        try:
            return await self._run_in_thread(self.clob_client.get_order, order_id)
        except Exception as e:
            logger.error(f"Failed to get order {order_id}: {e}")
            return None

    async def get_trades(
        self,
        token_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get trade history.

        Args:
            token_id: Optional token ID to filter
            limit: Maximum number of trades

        Returns:
            List of trades
        """
        try:
            trades = await self._run_in_thread(self.clob_client.get_trades, token_id, limit)
            logger.debug(f"Retrieved {len(trades)} trades")
            return trades
        except Exception as e:
            logger.error(f"Failed to get trades: {e}")
            return []

    async def get_order_book(self, token_id: str) -> Dict[str, Any]:
        """
        Get order book for a token.

        Args:
            token_id: Market token ID

        Returns:
            Order book data
        """
        try:
            return await self._run_in_thread(self.clob_client.get_order_book, token_id)
        except Exception as e:
            logger.error(f"Failed to get order book: {e}")
            return {}

    async def get_market_price(self, token_id: str) -> Dict[str, Any]:
        """
        Get current market price for a token.

        Args:
            token_id: Market token ID

        Returns:
            Price data
        """
        try:
            return await self._run_in_thread(self.clob_client.get_market_price, token_id)
        except Exception as e:
            logger.error(f"Failed to get market price: {e}")
            return {}

    async def deploy_safe_if_needed(self) -> bool:
        """
        Deploy Safe proxy wallet if not already deployed.

        Returns:
            True if deployment was needed or successful
        """
        if not self.config.use_gasless or not self.relayer_client:
            logger.debug("Gasless not enabled, skipping Safe deployment")
            return False

        try:
            response = await self._run_in_thread(
                self.relayer_client.deploy_safe,
                self.config.safe_address,
            )
            logger.info(f"Safe deployment initiated: {response}")
            return True
        except Exception as e:
            logger.warning(f"Safe deployment failed (may already be deployed): {e}")
            return False

    async def get_redeemable_positions(self) -> List[Dict[str, Any]]:
        """
        Get positions that can be redeemed (from resolved markets).

        Returns:
            List of redeemable positions with conditionId and other details
        """
        try:
            import httpx
            url = f"https://data-api.polymarket.com/positions"
            params = {
                "user": self.config.safe_address,
                "redeemable": "true",
            }
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params, timeout=10.0)
                response.raise_for_status()
                positions = response.json()
                logger.info(f"Found {len(positions)} redeemable positions")
                return positions
        except Exception as e:
            logger.error(f"Failed to get redeemable positions: {e}")
            return []

    async def redeem_position(self, condition_id: str) -> bool:
        """
        Redeem a resolved position to get USDC back.

        Args:
            condition_id: The condition ID of the resolved market

        Returns:
            True if redemption was successful
        """
        try:
            from web3 import Web3

            # Contract addresses on Polygon
            CTF_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
            USDC_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"

            # ABI for redeemPositions
            CTF_ABI = [{
                "name": "redeemPositions",
                "type": "function",
                "inputs": [
                    {"name": "collateralToken", "type": "address"},
                    {"name": "parentCollectionId", "type": "bytes32"},
                    {"name": "conditionId", "type": "bytes32"},
                    {"name": "indexSets", "type": "uint256[]"}
                ],
                "outputs": []
            }]

            w3 = Web3()
            ctf_contract = w3.eth.contract(
                address=Web3.to_checksum_address(CTF_ADDRESS),
                abi=CTF_ABI
            )

            # Encode the redeem call - redeem both outcomes [1, 2]
            parent_collection_id = bytes(32)  # bytes32(0)
            condition_id_bytes = bytes.fromhex(condition_id[2:] if condition_id.startswith("0x") else condition_id)
            index_sets = [1, 2]  # Redeem both YES and NO positions

            # Encode function call data using _encode_transaction_data (works with web3.py v6+)
            call_data = ctf_contract.functions.redeemPositions(
                Web3.to_checksum_address(USDC_ADDRESS),
                parent_collection_id,
                condition_id_bytes,
                index_sets
            )._encode_transaction_data()

            # Execute via relayer for gasless transaction
            if self.relayer_client:
                # Create SafeTransaction object for the official client
                redeem_tx = SafeTransaction(
                    to=CTF_ADDRESS,
                    operation=OperationType.Call,
                    data=call_data,
                    value="0"
                )

                # Execute transaction via official relayer client
                response = await self._run_in_thread(
                    self.relayer_client.execute,
                    [redeem_tx],
                    "Redeem positions"
                )

                logger.info(f"Redeem transaction submitted: ID={response.transaction_id}, Hash={response.transaction_hash}")

                # Wait for transaction to be mined on-chain
                logger.info("Waiting for transaction to be broadcast and mined...")
                result = await self._run_in_thread(
                    self.relayer_client.poll_until_state,
                    response.transaction_id,
                    [RelayerTransactionState.STATE_MINED.value, RelayerTransactionState.STATE_CONFIRMED.value],
                    RelayerTransactionState.STATE_FAILED.value,
                    max_polls=60,  # Wait up to 2 minutes
                    poll_frequency=2000  # Poll every 2 seconds
                )

                if result:
                    logger.info(f"Redemption successful! Final state: {result.get('state')}")
                    logger.info(f"Transaction hash: {result.get('transactionHash')}")
                    return True
                else:
                    logger.warning("Transaction failed or timed out")
                    return False
            else:
                logger.warning("Relayer client not available for gasless redeem")
                return False

        except Exception as e:
            logger.error(f"Failed to redeem position {condition_id}: {e}")
            return False

    async def auto_redeem_all(self) -> int:
        """
        Automatically redeem all redeemable positions.

        Returns:
            Number of positions successfully redeemed
        """
        positions = await self.get_redeemable_positions()
        if not positions:
            return 0

        redeemed = 0
        condition_ids_processed = set()

        for pos in positions:
            condition_id = pos.get("conditionId")
            if not condition_id or condition_id in condition_ids_processed:
                continue

            condition_ids_processed.add(condition_id)
            value = pos.get("currentValue", 0)
            logger.info(f"Redeeming position: {pos.get('title', 'Unknown')} (value: ${value:.2f})")

            if await self.redeem_position(condition_id):
                redeemed += 1

        logger.info(f"Redeemed {redeemed} positions")
        return redeemed

    def create_order_dict(
        self,
        token_id: str,
        price: float,
        size: float,
        side: str
    ) -> Dict[str, Any]:
        """
        Create an order dictionary for batch processing.

        Args:
            token_id: Market token ID
            price: Price per share
            size: Number of shares
            side: 'BUY' or 'SELL'

        Returns:
            Order dictionary
        """
        return {
            "token_id": token_id,
            "price": price,
            "size": size,
            "side": side.upper(),
        }


# Convenience function for quick initialization
def create_bot(
    config_path: str = "config.yaml",
    private_key: Optional[str] = None,
    encrypted_key_path: Optional[str] = None,
    password: Optional[str] = None,
    **kwargs
) -> TradingBot:
    """
    Create a TradingBot instance with common options.

    Args:
        config_path: Path to config file
        private_key: Private key (with 0x prefix)
        encrypted_key_path: Path to encrypted key file
        password: Password for encrypted key
        **kwargs: Additional arguments for TradingBot

    Returns:
        Configured TradingBot instance
    """
    return TradingBot(
        config_path=config_path,
        private_key=private_key,
        encrypted_key_path=encrypted_key_path,
        password=password,
        **kwargs
    )
