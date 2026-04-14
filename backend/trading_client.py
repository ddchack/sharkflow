"""
══════════════════════════════════════════════════════════════
Polymarket Bot - Trading Client
Developed by: Carlos David Donoso Cordero (ddchack)
══════════════════════════════════════════════════════════════

Wrapper around py-clob-client for executing trades.
Supports both manual and automated order placement.
"""

import os
import json
import time as _time_module
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Optional

TRADES_FILE = os.path.join(os.path.dirname(__file__), "trades.json")


@dataclass
class TradeRecord:
    """Record of an executed or attempted trade."""
    timestamp: str
    market_id: str
    question: str
    side: str
    token_id: str
    order_type: str  # "LIMIT" or "MARKET"
    price: float
    size_usd: float
    status: str  # "PENDING", "FILLED", "CANCELLED", "ERROR"
    order_id: str = ""
    error_msg: str = ""
    confidence: float = 0.0
    edge_pct: float = 0.0
    maker_order: bool = True   # True = post_only, 0% fees + rebate
    fee_paid_usd: float = 0.0  # Actual fee paid (negative = rebate received)
    category: str = "other"


class TradingClient:
    """
    Manages connection to Polymarket CLOB and executes trades.
    __author__ = "Carlos David Donoso Cordero (ddchack)"
    """

    def __init__(self, private_key: str = None, funder: str = None,
                 signature_type: int = 1, dry_run: bool = True):
        self.private_key = private_key or os.getenv("POLYMARKET_PRIVATE_KEY", "")
        self.funder = funder or os.getenv("POLYMARKET_FUNDER_ADDRESS", "")
        self.signature_type = signature_type
        self.dry_run = dry_run  # Safety: True = don't actually place orders
        self.clob_client = None          # cliente principal (balance queries, API auth)
        self.order_client = None          # cliente para firmar órdenes (puede ser sig_type=0 si sig_type=1 falla)
        self.is_authenticated = False
        self.trade_history: list[TradeRecord] = []
        self._daily_spent = 0.0
        self._daily_limit = float(os.getenv("MAX_CAPITAL_USD", "100"))
        self.validator = None            # TradeValidator — asignado desde api_server.py
        self._auth_timestamp: float = 0.0  # unix timestamp de la última autenticación exitosa
        self._load_trades()

    def _load_trades(self):
        """Carga el historial de trades desde trades.json y restaura el gasto diario.
        Soporta formato JSON array (legacy) y NDJSON (nuevo, O(1) append).
        """
        try:
            if not os.path.exists(TRADES_FILE):
                return
            with open(TRADES_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if not content:
                return
            # Detectar formato: JSON array legacy o NDJSON
            if content.startswith("["):
                data = json.loads(content)
            else:
                # NDJSON: una línea por trade
                data = []
                for line in content.splitlines():
                    line = line.strip()
                    if line:
                        try:
                            data.append(json.loads(line))
                        except Exception:
                            pass
            today = datetime.now(timezone.utc).date().isoformat()
            for item in data:
                record = TradeRecord(**{k: item[k] for k in item if k in TradeRecord.__dataclass_fields__})
                self.trade_history.append(record)
                # Restaurar gasto del día actual
                if record.status in ("FILLED", "DRY_RUN") and record.timestamp[:10] == today:
                    self._daily_spent += record.size_usd
            print(f"[TradingClient] Historial cargado: {len(self.trade_history)} trades "
                  f"(gastado hoy: ${self._daily_spent:.2f})")
        except Exception as e:
            print(f"[TradingClient] No se pudo cargar trades.json: {e}")

    def _append_trade(self, record: "TradeRecord"):
        """Agrega un trade al historial en memoria y lo persiste en trades.json.
        Usa NDJSON (una línea por trade) para O(1) append.
        Si el archivo existente es un JSON array legacy, lo migra a NDJSON primero.
        """
        self.trade_history.append(record)
        try:
            if not os.path.exists(TRADES_FILE):
                # Nuevo archivo: escribir directamente en NDJSON
                with open(TRADES_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
            else:
                # Detectar formato del archivo existente
                with open(TRADES_FILE, "r", encoding="utf-8") as f:
                    first_char = f.read(1)

                if first_char == "[":
                    # Formato legacy JSON array: leer, migrar a NDJSON
                    with open(TRADES_FILE, "r", encoding="utf-8") as f:
                        try:
                            existing = json.load(f)
                        except Exception:
                            existing = []
                    existing.append(asdict(record))
                    with open(TRADES_FILE, "w", encoding="utf-8") as f:
                        for entry in existing:
                            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    print("[TradingClient] trades.json migrado a formato NDJSON (O(1) append)")
                else:
                    # Ya es NDJSON: solo append
                    with open(TRADES_FILE, "a", encoding="utf-8") as f:
                        f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[TradingClient] Error guardando trade en JSON: {e}")

    def initialize(self) -> dict:
        """
        Initialize CLOB client and authenticate.

        Para wallets EOA (derived_addr == funder): fuerza sig_type=0 siempre.
          - sig_type=1 usa EIP-1271 que EOAs no implementan → invalid signature
          - sig_type=0 usa ECDSA estándar → correcto para EOA
        Para wallets proxy/gnosis-safe (derived_addr != funder): respeta sig_type del .env.
        """
        if not self.private_key or self.private_key.startswith("0x_YOUR"):
            return {
                "status": "not_configured",
                "message": "Private key not set. Export from https://reveal.magic.link/polymarket"
            }

        try:
            from py_clob_client.client import ClobClient

            # Detectar si es EOA (derived == funder) → forzar sig_type=0
            derived_addr = ""
            effective_sig_type = self.signature_type
            try:
                from eth_account import Account
                derived_addr = Account.from_key(self.private_key).address
                is_eoa = self.funder and derived_addr.lower() == self.funder.lower()
                if is_eoa:
                    effective_sig_type = 0  # EOA siempre usa tipo 0
                    if self.signature_type != 0:
                        print(f"[TradingClient] EOA detectado → forzando sig_type=0 "
                              f"(sig_type=1 usa EIP-1271 que EOAs no implementan)")
                    else:
                        print(f"[TradingClient] EOA detectado, sig_type=0 correcto.")
                else:
                    print(f"[TradingClient] Proxy/GnosisSafe detectado, usando sig_type={effective_sig_type}")
            except Exception:
                pass

            # Un único cliente con el sig_type efectivo (0 para EOA, config para proxy)
            self.clob_client = ClobClient(
                host="https://clob.polymarket.com",
                key=self.private_key,
                chain_id=137,
                signature_type=effective_sig_type,
                funder=self.funder
            )
            creds = self.clob_client.create_or_derive_api_creds()
            self.clob_client.set_api_creds(creds)
            self.order_client = self.clob_client
            self.is_authenticated = True
            self._auth_timestamp = _time_module.time()

            # Intentar configurar allowances USDC (requiere MATIC para gas)
            allowance_status = self._setup_allowances()

            print(f"[TradingClient] Autenticado. effective_sig_type={effective_sig_type}, "
                  f"funder={self.funder[:10] if self.funder else '?'}..., "
                  f"allowances={allowance_status}")

            return {
                "status": "authenticated",
                "message": "Successfully connected to Polymarket CLOB",
                "funder": self.funder,
                "derived_address": derived_addr,
                "signature_type": self.signature_type,
                "api_key": creds.api_key if hasattr(creds, 'api_key') else "***",
                "allowances": allowance_status,
            }

        except ImportError:
            return {
                "status": "missing_dependency",
                "message": "py-clob-client not installed. Run: pip install py-clob-client"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Authentication failed: {str(e)}"
            }

    def _setup_allowances(self) -> str:
        """
        Configura los permisos (ERC-20 approve) necesarios para operar en Polymarket CLOB.
        Intenta:
          1. update_allowances() del ClobClient (versiones nuevas de py-clob-client)
          2. Fallback manual usando web3: approve USDC + setApprovalForAll CTF
        Retorna un string describiendo el resultado.
        """
        # ── Intento 1: método nativo de py-clob-client ────────────────────
        try:
            self.clob_client.update_allowances()
            return "set via update_allowances()"
        except AttributeError:
            pass  # versión antigua, seguir con fallback
        except Exception as e1:
            print(f"[TradingClient] update_allowances() error: {e1}")

        # ── Intento 2: web3 manual ────────────────────────────────────────
        try:
            from web3 import Web3
            from eth_account import Account

            # Obtener addresses desde py_clob_client si existen
            try:
                from py_clob_client.constants import (
                    EXCHANGE_ADDRESSES, NEG_RISK_EXCHANGE_ADDRESSES,
                    COLLATERAL_TOKEN_ADDRESSES, CONDITIONAL_TOKENS_ADDRESSES,
                )
                EXCHANGE     = Web3.to_checksum_address(EXCHANGE_ADDRESSES.get(137, ""))
                NEG_EXCHANGE = Web3.to_checksum_address(NEG_RISK_EXCHANGE_ADDRESSES.get(137, ""))
                USDC         = Web3.to_checksum_address(COLLATERAL_TOKEN_ADDRESSES.get(137, ""))
                CTF          = Web3.to_checksum_address(CONDITIONAL_TOKENS_ADDRESSES.get(137, ""))
            except Exception:
                # Direcciones conocidas de Polymarket en Polygon mainnet
                EXCHANGE     = Web3.to_checksum_address("0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E")
                NEG_EXCHANGE = ""  # Se obtiene de py_clob_client.constants; sin fallback hardcodeado
                USDC         = Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")
                CTF          = Web3.to_checksum_address("0x4D97DCd97eC945f40cF65F87097ACe5EA0476045")

            # Conectar a Polygon RPC
            RPC_URLS = [
                "https://polygon-rpc.com",
                "https://rpc-mainnet.matic.network",
                "https://matic-mainnet.chainstacklabs.com",
            ]
            w3 = None
            for rpc in RPC_URLS:
                try:
                    _w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 10}))
                    if _w3.is_connected():
                        w3 = _w3
                        break
                except Exception:
                    continue
            if not w3:
                return "skipped: no RPC disponible"

            account = Account.from_key(self.private_key)
            MAX_UINT = 2**256 - 1

            # ABI mínimo para ERC-20 approve + allowance
            ERC20_ABI = [
                {"inputs": [{"name": "spender", "type": "address"}, {"name": "amount", "type": "uint256"}],
                 "name": "approve", "outputs": [{"name": "", "type": "bool"}],
                 "stateMutability": "nonpayable", "type": "function"},
                {"inputs": [{"name": "owner", "type": "address"}, {"name": "spender", "type": "address"}],
                 "name": "allowance", "outputs": [{"name": "", "type": "uint256"}],
                 "stateMutability": "view", "type": "function"},
            ]
            # ABI para ERC-1155 setApprovalForAll (CTF tokens)
            ERC1155_ABI = [
                {"inputs": [{"name": "operator", "type": "address"}, {"name": "approved", "type": "bool"}],
                 "name": "setApprovalForAll", "outputs": [],
                 "stateMutability": "nonpayable", "type": "function"},
                {"inputs": [{"name": "account", "type": "address"}, {"name": "operator", "type": "address"}],
                 "name": "isApprovedForAll", "outputs": [{"name": "", "type": "bool"}],
                 "stateMutability": "view", "type": "function"},
            ]

            usdc_contract = w3.eth.contract(address=USDC, abi=ERC20_ABI)
            ctf_contract  = w3.eth.contract(address=CTF,  abi=ERC1155_ABI)

            nonce    = w3.eth.get_transaction_count(account.address)
            gas_price = w3.eth.gas_price
            base_tx  = {"from": account.address, "gasPrice": gas_price, "chainId": 137}

            txs_sent = []

            # 1) USDC.approve(EXCHANGE, MAX_UINT)
            current_allowance = usdc_contract.functions.allowance(account.address, EXCHANGE).call()
            if current_allowance < 10**6 * 100:  # < $100 → renovar
                tx = usdc_contract.functions.approve(EXCHANGE, MAX_UINT).build_transaction(
                    {**base_tx, "nonce": nonce, "gas": 80000})
                signed = account.sign_transaction(tx)
                # Compatibilidad web3 <6 (rawTransaction) y >=6 (raw_transaction)
                raw_tx = getattr(signed, "raw_transaction", None) or getattr(signed, "rawTransaction", None)
                tx_hash = w3.eth.send_raw_transaction(raw_tx)
                txs_sent.append(f"USDC_approve:{tx_hash.hex()[:12]}")
                nonce += 1

            # 2) CTF.setApprovalForAll(EXCHANGE, True)
            if not ctf_contract.functions.isApprovedForAll(account.address, EXCHANGE).call():
                tx = ctf_contract.functions.setApprovalForAll(EXCHANGE, True).build_transaction(
                    {**base_tx, "nonce": nonce, "gas": 80000})
                signed = account.sign_transaction(tx)
                # Compatibilidad web3 <6 (rawTransaction) y >=6 (raw_transaction)
                raw_tx = getattr(signed, "raw_transaction", None) or getattr(signed, "rawTransaction", None)
                tx_hash = w3.eth.send_raw_transaction(raw_tx)
                txs_sent.append(f"CTF_approve:{tx_hash.hex()[:12]}")
                nonce += 1

            # 3) CTF.setApprovalForAll(NEG_EXCHANGE, True) si la dirección es válida
            if NEG_EXCHANGE and len(NEG_EXCHANGE) == 42:
                try:
                    if not ctf_contract.functions.isApprovedForAll(account.address, NEG_EXCHANGE).call():
                        tx = ctf_contract.functions.setApprovalForAll(NEG_EXCHANGE, True).build_transaction(
                            {**base_tx, "nonce": nonce, "gas": 80000})
                        signed = account.sign_transaction(tx)
                        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
                        txs_sent.append(f"NEG_approve:{tx_hash.hex()[:12]}")
                        nonce += 1
                except Exception:
                    pass

            if txs_sent:
                print(f"[TradingClient] Allowances configurados vía web3: {txs_sent}")
                return f"set via web3: {', '.join(txs_sent)}"
            else:
                return "already_set (allowances existentes)"

        except Exception as e2:
            print(f"[TradingClient] _setup_allowances web3 error: {e2}")
            return f"skipped: {str(e2)[:100]}"

    # ─────────────────────────────────────────────────────────
    # ORDER PLACEMENT
    # ─────────────────────────────────────────────────────────

    def place_limit_order(self, token_id: str, price: float, size: float,
                           side: str, market_id: str = "",
                           question: str = "", confidence: float = 0,
                           edge_pct: float = 0,
                           market_data: Optional[dict] = None,
                           fok: bool = False,
                           post_only: bool = True) -> TradeRecord:
        """
        Place a GTC limit order.
        
        Args:
            token_id: The CLOB token ID (YES or NO token)
            price: Limit price (0.01 - 0.99)
            size: Number of shares
            side: "BUY" or "SELL"
        """
        # MAKER-ONLY por defecto (post_only=True):
        # - Si la orden se ejecutaría inmediatamente como TAKER → se cancela
        # - Si se coloca como MAKER → 0% fees + rebate 20-25%
        # - Cambio de impacto: fee crypto pasa de 7.2% a 0% + rebate
        record = TradeRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            market_id=market_id,
            question=question,
            side=side,
            token_id=token_id,
            order_type="LIMIT",
            price=price,
            size_usd=round(price * size, 2),
            status="PENDING",
            confidence=confidence,
            edge_pct=edge_pct,
            maker_order=post_only,
        )

        # ── TradeValidator: safety net en place_limit_order ──────────────
        if self.validator is not None:
            # Usar precios reales del mercado si se proveen; si no, usar precio límite
            val_data = market_data if market_data else {"yes_price": price, "no_price": 1.0 - price}
            # outcome: "YES" al comprar (BUY), "NO" al vender (SELL)
            val_outcome = "YES" if side.upper() == "BUY" else "NO"
            is_valid, block_reason = self.validator.validate_trade(
                market_id=market_id,
                outcome=val_outcome,
                amount=record.size_usd,
                kelly_amount=record.size_usd,
                market_data=val_data,
            )
            if not is_valid:
                record.status = "BLOCKED"
                record.error_msg = f"TradeValidator: {block_reason}"
                self._append_trade(record)
                return record

        # Daily limit check
        if self._daily_spent + record.size_usd > self._daily_limit:
            record.status = "REJECTED"
            record.error_msg = f"Daily limit ${self._daily_limit} would be exceeded"
            self._append_trade(record)
            return record

        if self.dry_run:
            record.status = "DRY_RUN"
            record.order_id = f"dry_{datetime.now().strftime('%H%M%S')}"
            self._daily_spent += record.size_usd
            self._append_trade(record)
            return record

        if not self.is_authenticated:
            record.status = "ERROR"
            record.error_msg = "Not authenticated. Call initialize() first."
            self._append_trade(record)
            return record

        try:
            from py_clob_client.clob_types import OrderArgs, OrderType
            from py_clob_client.order_builder.constants import BUY, SELL

            # Polymarket CLOB: precio max 2 decimales, size max 4 decimales.
            # makerAmount = size × price debe ser exactamente 2 decimales.
            # El caller garantiza esto usando tokens enteros (sin decimales) en contrarian.
            price = round(price, 2)
            size  = round(size,  4)

            order_side = BUY if side.upper() == "BUY" else SELL
            order_args = OrderArgs(
                token_id=token_id,
                price=price,
                size=size,
                side=order_side,
            )

            signed_order = self.clob_client.create_order(order_args)
            # post_only=True → maker order (fee rebate). Si fok=True se ignora post_only.
            if fok:
                order_type = OrderType.FOK
            elif post_only:
                order_type = OrderType.GTD  # GTD con precio límite actúa como post-only en Polymarket CLOB
            else:
                order_type = OrderType.GTC
            resp = self.clob_client.post_order(signed_order, order_type)

            if resp.get("success"):
                record.status = "FILLED"
                record.order_id = resp.get("orderID", "")
                self._daily_spent += record.size_usd
            else:
                # FOK sin fill o GTC rechazado: distinguir de errores de red/auth
                record.status = "FOK_CANCELLED" if fok else "ERROR"
                record.order_id = resp.get("orderID", "")
                record.error_msg = resp.get("errorMsg", "") or resp.get("message", "") or "order not filled"
                # FOK cancelado = orden nunca ejecutada → revertir dedup para permitir reintento
                if fok and self.validator is not None:
                    self.validator.cancel_dedup(market_id)

        except Exception as e:
            err_str = str(e)
            # Polymarket lanza excepción para FOK sin fill (no devuelve success=False)
            _is_fok_kill = fok and (
                "couldn't be fully filled" in err_str or
                "FOK orders are fully filled or killed" in err_str or
                "fully filled or killed" in err_str
            )
            record.status = "FOK_CANCELLED" if _is_fok_kill else "ERROR"
            record.error_msg = err_str
            if self.validator is not None and (fok or _is_fok_kill):
                self.validator.cancel_dedup(market_id)

        self._append_trade(record)
        return record

    def place_market_order(self, token_id: str, amount_usd: float,
                            side: str, market_id: str = "",
                            question: str = "", confidence: float = 0,
                            edge_pct: float = 0,
                            market_data: Optional[dict] = None) -> TradeRecord:
        """
        Place a FOK (Fill or Kill) market order.
        
        Args:
            token_id: The CLOB token ID
            amount_usd: Amount in USD to spend
            side: "BUY" or "SELL"
        """
        record = TradeRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            market_id=market_id,
            question=question,
            side=side,
            token_id=token_id,
            order_type="MARKET",
            price=0.0,
            size_usd=amount_usd,
            status="PENDING",
            confidence=confidence,
            edge_pct=edge_pct,
        )

        # ── TradeValidator: safety net en place_market_order ─────────────
        if self.validator is not None:
            # Normalizar outcome: token_id es un hash hex, no contiene "yes"/"no".
            # Determinar YES/NO a partir del parámetro side.
            if side in ("YES", "NO"):
                outcome = side
            elif side.upper() == "BUY":
                outcome = "YES"
            elif side.upper() == "SELL":
                outcome = "NO"
            else:
                outcome = "YES"  # default conservador
            is_valid, block_reason = self.validator.validate_trade(
                market_id=market_id,
                outcome=outcome,
                amount=amount_usd,
                kelly_amount=amount_usd,
                market_data=market_data or {},
            )
            if not is_valid:
                record.status = "BLOCKED"
                record.error_msg = f"TradeValidator: {block_reason}"
                self._append_trade(record)
                return record

        if self._daily_spent + amount_usd > self._daily_limit:
            record.status = "REJECTED"
            record.error_msg = f"Daily limit ${self._daily_limit} would be exceeded"
            self._append_trade(record)
            return record

        if self.dry_run:
            record.status = "DRY_RUN"
            record.order_id = f"dry_mkt_{datetime.now().strftime('%H%M%S')}"
            self._daily_spent += amount_usd
            self._append_trade(record)
            return record

        if not self.is_authenticated:
            record.status = "ERROR"
            record.error_msg = "Not authenticated"
            self._append_trade(record)
            return record

        client = self.clob_client

        try:
            from py_clob_client.clob_types import MarketOrderArgs, OrderType
            from py_clob_client.order_builder.constants import BUY, SELL

            order_side = BUY if side.upper() == "BUY" else SELL
            mo = MarketOrderArgs(
                token_id=token_id,
                amount=amount_usd,
                side=order_side,
            )

            signed = client.create_market_order(mo)
            resp = client.post_order(signed, OrderType.FOK)

            record.status = "FILLED" if resp.get("success") else "ERROR"
            record.order_id = resp.get("orderID", "")
            record.error_msg = resp.get("errorMsg", "") or resp.get("error", "")
            record.price = float(resp.get("price", 0))
            if record.status == "ERROR":
                print(f"[TradingClient] CLOB rechazó orden: {resp}")
            self._daily_spent += amount_usd

        except Exception as e:
            record.status = "ERROR"
            record.error_msg = str(e)
            print(f"[TradingClient] ERROR place_market_order: {e}")

        self._append_trade(record)
        return record

    # ─────────────────────────────────────────────────────────
    # ORDER MANAGEMENT
    # ─────────────────────────────────────────────────────────

    def cancel_order(self, order_id: str) -> dict:
        """Cancel an open order."""
        if self.dry_run:
            return {"status": "DRY_RUN", "order_id": order_id}

        if not self.is_authenticated:
            return {"status": "error", "message": "Not authenticated"}

        try:
            resp = self.clob_client.cancel(order_id)
            return {"status": "cancelled", "response": resp}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def cancel_all_orders(self) -> dict:
        """Cancel all open orders."""
        if self.dry_run:
            return {"status": "DRY_RUN", "message": "Would cancel all orders"}

        if not self.is_authenticated:
            return {"status": "error", "message": "Not authenticated"}

        try:
            resp = self.clob_client.cancel_all()
            return {"status": "cancelled_all", "response": resp}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_open_orders(self, market_id: str = None) -> list:
        """Get current open orders."""
        if not self.is_authenticated:
            return []

        try:
            if market_id:
                from py_clob_client.clob_types import OpenOrderParams
                params = OpenOrderParams(market=market_id)
                return self.clob_client.get_orders(params)
            return self.clob_client.get_orders()
        except Exception as e:
            print(f"[Trading] Error getting orders: {e}")
            return []

    # ─────────────────────────────────────────────────────────
    # PORTFOLIO & STATS
    # ─────────────────────────────────────────────────────────

    def get_trade_summary(self) -> dict:
        """Summary of all trades in this session."""
        filled = [t for t in self.trade_history if t.status in ("FILLED", "DRY_RUN")]
        errors = [t for t in self.trade_history if t.status == "ERROR"]

        total_wagered = sum(t.size_usd for t in filled)

        return {
            "total_trades": len(self.trade_history),
            "filled": len(filled),
            "errors": len(errors),
            "total_wagered_usd": round(total_wagered, 2),
            "daily_spent": round(self._daily_spent, 2),
            "daily_remaining": round(self._daily_limit - self._daily_spent, 2),
            "dry_run": self.dry_run,
            "is_authenticated": self.is_authenticated,
            "history": [
                {
                    "timestamp": t.timestamp,
                    "question": t.question[:80],
                    "side": t.side,
                    "type": t.order_type,
                    "price": t.price,
                    "size_usd": t.size_usd,
                    "status": t.status,
                    "confidence": t.confidence,
                    "order_id": t.order_id,
                    "market_id": t.market_id,
                }
                for t in self.trade_history[-50:]  # Last 50 trades
            ]
        }

    def reset_daily_limit(self):
        """Reset daily spending counter (call at midnight)."""
        self._daily_spent = 0.0

    def get_filled_positions_today(self) -> dict:
        """
        Retorna {market_id: size_usd} de trades FILLED del día de hoy.
        Usado para restaurar el position tracker del RiskManager tras un reinicio,
        evitando que el bot re-apueste en mercados donde ya tiene posición abierta.
        """
        today = datetime.now(timezone.utc).date().isoformat()
        positions: dict = {}
        for t in self.trade_history:
            if t.status == "FILLED" and t.market_id and t.timestamp[:10] == today:
                positions[t.market_id] = positions.get(t.market_id, 0.0) + t.size_usd
        return positions

    # ─────────────────────────────────────────────────────────
    # BALANCE REAL EN POLYGON / CLOB
    # ─────────────────────────────────────────────────────────

    def _get_data_api_balance(self) -> float | None:
        """
        Consulta el balance USDC usando el Data API público de Polymarket.
        Solo requiere el funder address — sin private key ni HMAC.

        Prueba varios endpoints de la API pública de Polymarket.
        Retorna el balance en USDC, o None si todos fallan.
        __author__ = "Carlos David Donoso Cordero (ddchack)"
        """
        if not self.funder or self.funder.startswith("0x_YOUR"):
            return None

        import urllib.request as _req

        address = self.funder.lower()
        headers = {"Accept": "application/json", "User-Agent": "SharkFlow/4.0"}

        # Nota: data-api.polymarket.com/value devuelve el valor de posiciones abiertas,
        # NO el saldo USDC en efectivo. El efectivo sin desplegar en el CLOB solo es
        # accesible con autenticación L2 (requiere Private Key).
        # Este método queda como placeholder para cuando Polymarket exponga un endpoint público.
        return None

    def debug_data_api(self) -> dict:
        """Devuelve diagnóstico del balance: qué endpoints respondieron y qué contienen."""
        if not self.funder or self.funder.startswith("0x_YOUR"):
            return {"error": "Funder address no configurado"}

        import urllib.request as _req
        address = self.funder.lower()
        headers = {"Accept": "application/json", "User-Agent": "SharkFlow/4.0"}
        results = {}

        endpoints = [
            f"https://data-api.polymarket.com/portfolio?user={address}&limit=3",
            f"https://data-api.polymarket.com/value?user={address}",
        ]
        for url in endpoints:
            try:
                req = _req.Request(url, headers=headers, method="GET")
                with _req.urlopen(req, timeout=10) as resp:
                    results[url] = json.loads(resp.read().decode("utf-8"))
            except Exception as e:
                results[url] = {"error": str(e)}

        return {
            "address": address,
            "raw": results,
            "nota": ("El saldo USDC en efectivo del CLOB de Polymarket solo es accesible "
                     "con autenticación L2 (Private Key). Los endpoints públicos solo muestran "
                     "el valor de posiciones abiertas, no el efectivo sin desplegar.")
        }

    def get_wallet_balance(self) -> dict:
        """
        Consulta el balance real de USDC del funder address en Polymarket/Polygon.

        Intenta en orden:
          1. py_clob_client.get_balance_allowance() — balance DENTRO del sistema Polymarket.
             El saldo que muestra la web de Polymarket está en sus contratos internos,
             NO en la wallet on-chain. Requiere private key para autenticar.
             Si tiene private_key pero aún no está autenticado, hace auto-init.
          2. HMAC directo con api_key/api_secret/api_passphrase (sin private key).
             Obtener en polymarket.com → Perfil → Settings → API Keys.
          3. Polygon RPC + USDC.e (0x2791Bc…) — balance on-chain directo
          4. Polygon RPC + USDC nativo (0x3c499c…)

        Devuelve:
          {
            "usdc": float,
            "source": str,  # "clob" | "polygon_rpc (USDC.e)" | "not_configured" | ...
            "address": str,
            "error": str | None,
            "needs_private_key": bool  # True si solo tenemos funder address pero no PK
          }
        """
        address = self.funder.strip() if self.funder else ""
        has_pk  = bool(self.private_key and not self.private_key.startswith("0x_YOUR"))

        if not address or address.startswith("0x_YOUR"):
            return {"usdc": 0.0, "source": "not_configured",
                    "address": "", "error": "Funder address no configurado",
                    "needs_private_key": False}

        # ── Intento 1: CLOB client — balance INTERNO de Polymarket ────────
        # El saldo que muestra polymarket.com está aquí, no on-chain.
        # Intentar auto-inicializar si tenemos PK pero no estamos autenticados.
        if has_pk and not self.is_authenticated:
            try:
                self.initialize()
            except Exception:
                pass

        if self.is_authenticated and self.clob_client:
            try:
                from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
                # AssetType.COLLATERAL = USDC disponible para operar en el CLOB
                params   = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
                bal_data = self.clob_client.get_balance_allowance(params)
                if isinstance(bal_data, dict):
                    raw  = float(bal_data.get("balance", "0") or "0")
                    usdc = raw / 1_000_000   # USDC tiene 6 decimales
                    if usdc > 0:
                        return {"usdc": round(usdc, 4),
                                "source": "clob", "address": address,
                                "error": None, "needs_private_key": False}
                    # Si CLOB devuelve 0, continuar a RPC para verificar balance on-chain
                    # (puede pasar si el USDC está en wallet directamente, no en CTF)
            except Exception:
                pass  # fallback a RPC

        # ── Intento 2: Data API público de Polymarket (solo address) ────
        data_bal = self._get_data_api_balance()
        if data_bal is not None:
            return {"usdc": data_bal, "source": "polymarket_data_api",
                    "address": address, "error": None,
                    "needs_private_key": False}

        # ── Intento 3 y 4: Polygon JSON-RPC ──────────────────────────────
        import urllib.request
        import struct

        # ERC-20 balanceOf(address) → selector = 0x70a08231
        addr_padded = address.lower().replace("0x", "").zfill(64)
        data = "0x70a08231" + addr_padded

        rpc_payload = json.dumps({
            "jsonrpc": "2.0", "method": "eth_call",
            "params": [{"to": "<TOKEN>", "data": data}, "latest"],
            "id": 1
        }).encode()

        usdc_contracts = [
            ("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174", "USDC.e"),  # bridged
            ("0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359", "USDC"),    # native
        ]
        rpc_endpoints = [
            "https://rpc.ankr.com/polygon",
            "https://gateway.tenderly.co/public/polygon",
            "https://polygon-rpc.com",
        ]

        best_usdc = 0.0
        best_source = None
        any_rpc_success = False
        last_queried_name = None

        for contract_addr, contract_name in usdc_contracts:
            payload = rpc_payload.replace(b"<TOKEN>", contract_addr.encode())
            for rpc_url in rpc_endpoints:
                try:
                    req = urllib.request.Request(
                        rpc_url,
                        data=payload,
                        headers={"Content-Type": "application/json",
                                 "Accept": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=6) as resp:
                        result = json.loads(resp.read().decode("utf-8"))
                    hex_val = result.get("result", "0x0") or "0x0"
                    raw = int(hex_val, 16) if hex_val not in ("0x", "") else 0
                    usdc = raw / 1_000_000  # USDC tiene 6 decimales
                    any_rpc_success = True
                    last_queried_name = contract_name
                    if usdc > best_usdc:
                        best_usdc = usdc
                        best_source = contract_name
                    if usdc > 0:
                        break  # Encontramos balance en este contrato, suficiente
                except Exception:
                    continue

        # Nota: el balance on-chain puede ser 0 aunque Polymarket muestre saldo,
        # porque los fondos están en los contratos CTF de Polymarket (no en la wallet EOA).
        # Para ver el balance real de Polymarket se necesita la private_key.
        needs_pk = not has_pk

        if any_rpc_success:
            source_name = best_source or last_queried_name
            note = None if best_usdc > 0 else (
                "Balance on-chain 0. Tu saldo de Polymarket está en sus contratos internos. "
                "Configura la Private Key para verlo." if needs_pk else
                "Fondos dentro de Polymarket. Ejecuta /api/auth/connect para sincronizar."
            )
            return {
                "usdc":              round(best_usdc, 4),
                "source":            f"polygon_rpc ({source_name})",
                "address":           address,
                "error":             note,
                "needs_private_key": needs_pk,
            }

        return {
            "usdc":              0.0,
            "source":            "rpc_failed",
            "address":           address,
            "error":             "No se pudo consultar el balance via RPC",
            "needs_private_key": needs_pk,
        }

    # ─────────────────────────────────────────────────────────
    # AUTO-RECONNECT
    # ─────────────────────────────────────────────────────────

    def ensure_connected(self, max_age_hours: float = 3.0, max_retries: int = 5) -> bool:
        """Verifica y renueva conexión CLOB con retry exponential backoff."""
        import time as _time_local
        if not self.private_key or self.private_key.startswith("0x_YOUR"):
            return False
        if not getattr(self, 'funder', None):
            return False

        ts = getattr(self, '_auth_timestamp', 0)
        age_hours = (_time_local.time() - ts) / 3600.0 if ts > 0 else 999.0

        if getattr(self, 'is_authenticated', False) and age_hours < max_age_hours:
            return True

        reason = f"age={age_hours:.1f}h>{max_age_hours}h" if getattr(self, 'is_authenticated', False) else "not_authenticated"

        for attempt in range(1, max_retries + 1):
            print(f"[TradingClient] Reconectando CLOB intento {attempt}/{max_retries} ({reason})...")
            try:
                result = self.initialize()
                if isinstance(result, dict) and result.get("status") == "authenticated":
                    print(f"[TradingClient] Reconexion exitosa en intento {attempt}.")
                    return True
                msg = result.get("message", "?") if isinstance(result, dict) else str(result)
                print(f"[TradingClient] Intento {attempt} fallido: {msg}")
            except Exception as e:
                print(f"[TradingClient] Intento {attempt} excepcion: {e}")

            if attempt < max_retries:
                wait = min(2 ** attempt, 60)
                print(f"[TradingClient] Esperando {wait}s antes de retry...")
                _time_local.sleep(wait)

        print(f"[TradingClient] FALLO: {max_retries} intentos agotados. Conexion CLOB perdida.")
        self.is_authenticated = False
        return False
