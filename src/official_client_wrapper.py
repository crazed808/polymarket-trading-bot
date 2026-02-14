"""
Wrapper to use official Polymarket SDK with the existing bot structure
"""
from typing import Optional, Dict, Any, List
from py_clob_client.client import ClobClient as OfficialClobClient
from py_clob_client.clob_types import OrderArgs, OrderType
from .config import BuilderConfig

class ClobClient:
    """Wrapper around official py-clob-client"""
    
    def __init__(
        self,
        host: str,
        chain_id: int,
        signature_type: int,
        funder: str,
        private_key: str,
        api_creds=None,
        builder_creds: Optional[BuilderConfig] = None,
    ):
        # Remove 0x prefix if present
        if private_key.startswith('0x'):
            private_key = private_key[2:]
            
        self.client = OfficialClobClient(
            host=host,
            key=private_key,
            chain_id=chain_id,
            signature_type=signature_type,
            funder=funder
        )
        
        # Derive and set API credentials
        if api_creds is None:
            creds = self.client.create_or_derive_api_creds()
            self.client.set_api_creds(creds)
        else:
            self.client.set_api_creds(api_creds)
    
    def get_open_orders(self) -> List[Dict[str, Any]]:
        """Get all open orders"""
        return self.client.get_orders()
    
    def get_trades(self, token_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get trade history"""
        return self.client.get_trades()
    
    def get_order_book(self, token_id: str) -> Dict[str, Any]:
        """Get order book for a token"""
        return self.client.get_order_book(token_id)
    
    def get_market_price(self, token_id: str) -> Dict[str, Any]:
        """Get current market price"""
        return self.client.get_price(token_id)
    
    def post_order(self, signed_order: Dict[str, Any], order_type: str = "GTC") -> Dict[str, Any]:
        """Submit a signed order"""
        # Convert to official SDK format
        order_args = OrderArgs(
            price=float(signed_order['order']['price']),
            size=float(signed_order['order']['size']),
            side=signed_order['order']['side'],
            token_id=signed_order['order']['tokenId'],
        )
        return self.client.create_order(order_args)
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an order"""
        return self.client.cancel(order_id)
    
    def cancel_all_orders(self) -> Dict[str, Any]:
        """Cancel all open orders"""
        return self.client.cancel_all()
