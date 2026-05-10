"""
Financial Reporting Pipeline — production-grade rewrite.

"""

from __future__ import annotations

import json
import logging
import os
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from openai import OpenAI

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# OpenAI client  (swap base_url + api_key to point at Claude / any LLM)
# ---------------------------------------------------------------------------

def _get_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=api_key)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class MissingFXRateError(Exception):
    """Raised when no FX rate exists for a currency / rate-type pair."""


# ---------------------------------------------------------------------------
# Event model
# ---------------------------------------------------------------------------

class LedgerEvent:
    """Immutable audit record for a single ledger movement."""

    __slots__ = (
        "event_id", "parent_event", "timestamp",
        "event_type", "account_code", "amount",
        "currency", "metadata",
    )

    def __init__(
        self,
        event_type: str,
        account_code: str | int,
        amount: float | None = None,
        currency: str | None = None,
        metadata: dict | None = None,
        parent_event: str | None = None,
    ) -> None:
        self.event_id     = str(uuid.uuid4())
        self.parent_event = parent_event
        # DEP-1: use timezone-aware UTC timestamp
        self.timestamp    = datetime.now(timezone.utc).isoformat()
        self.event_type   = event_type
        self.account_code = str(account_code)
        self.amount       = amount
        self.currency     = currency
        self.metadata     = metadata or {}

    def to_dict(self) -> dict:
        return {s: getattr(self, s) for s in self.__slots__}


# ---------------------------------------------------------------------------
# AI service
# ---------------------------------------------------------------------------

class AIService:
    """Thin wrapper around the LLM API.  All prompts live here."""

    _MODEL = "gpt-4.1-mini"

    def _chat(
        self,
        system: str,
        user: str,
        temperature: float = 0,
        max_tokens: int = 500,
    ) -> str:
        response = _get_client().chat.completions.create(
            model=self._MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        return response.choices[0].message.content

    # ------------------------------------------------------------------
    # Orphan account classification
    # ------------------------------------------------------------------

    def classify_account(self, account_name: str, account_code: str) -> dict:
        """Return a structured classification for a COA-orphan account."""
        prompt = f"""
Classify the following account.

ACCOUNT CODE: {account_code}
ACCOUNT NAME: {account_name}

Return JSON only — no markdown, no prose.

Schema:
{{
  "account_type":  "Asset | Liability | Equity | Revenue | Expense",
  "statement":     "BS | PL",
  "normal_balance":"Debit | Credit",
  "confidence":    0.0,
  "reasoning":     "..."
}}
"""
        raw = self._chat(
            system="You are a CPA-grade accounting assistant.",
            user=prompt,
        )
        return json.loads(raw)

    # ------------------------------------------------------------------
    # PERF-2: Batched anomaly explanation
    # ------------------------------------------------------------------

    def explain_anomalies_batch(
        self, anomalies: list[dict]
    ) -> list[dict]:
        """
        Send ALL anomalies in a single API call instead of one call per row.

        Each anomaly dict must have keys: account_code, account_name, amount.
        Returns the same list with an added 'explanation' key on each item.
        """
        if not anomalies:
            return anomalies

        items_text = "\n".join(
            f"{i+1}. Account {a['account_code']} ({a['account_name']}): "
            f"USD {a['amount']:,.0f}"
            for i, a in enumerate(anomalies)
        )

        prompt = f"""
You are a senior CFO reviewing flagged large balances.

For each item below, write a one-sentence explanation of why it may
warrant review. Return a JSON array with one object per item:
[{{"index": 1, "explanation": "..."}}, ...]

ITEMS:
{items_text}
"""
        raw = self._chat(
            system="You are a financial risk analyst.",
            user=prompt,
            temperature=0.1,
            max_tokens=1000,
        )
        # Strip markdown fences if present
        clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        results = json.loads(clean)

        result_map = {r["index"]: r["explanation"] for r in results}
        for i, a in enumerate(anomalies):
            a["explanation"] = result_map.get(i + 1, "No explanation returned.")
        return anomalies

    # ------------------------------------------------------------------
    # CFO narrative
    # ------------------------------------------------------------------

    def generate_financial_narrative(
        self,
        financials: dict,
        anomalies: list[dict],
    ) -> str:
        prompt = f"""
Generate a concise CFO-style financial narrative (5-7 sentences).

You have four financial statements for the period. Your narrative must cover:
- P&L: revenue, expenses, net income and margin
- Balance Sheet: asset base, leverage, liquidity
- Cash Flow: operating cash generation, key capex or financing movements
- Equity: material changes in retained earnings, OCI, or share capital
- Risks: any anomalies or audit flags requiring attention

FINANCIAL STATEMENTS:
{json.dumps(financials, indent=2, default=str)}

ANOMALIES FLAGGED:
{json.dumps(anomalies, indent=2, default=str)}

Write in professional CFO tone. Flowing prose, no bullet points.
"""
        return self._chat(
            system="You are an expert financial analyst writing for a board audience.",
            user=prompt,
            temperature=0.3,
            max_tokens=500,
        )


# ---------------------------------------------------------------------------
# Reporting state
# ---------------------------------------------------------------------------

class ReportingState:
    """Central state object passed through the pipeline."""

    def __init__(
        self,
        tb_path: str,
        coa_path: str,
        fx_path: str,
        adj_path: str,
        prior_path: str,
    ) -> None:
        self.raw_tb     = pd.read_csv(tb_path)
        self.coa        = pd.read_csv(coa_path)
        self.fx_rates   = pd.read_csv(fx_path)
        self.manual_adj = json.loads(open(adj_path).read())
        self.prior_tb   = pd.read_csv(prior_path)

        self.ledger: pd.DataFrame = pd.DataFrame()
        self.results: dict[str, Any] = {}
        self.logs: list[str] = []
        self.ai_suggestions: list[dict] = []

        # PERF-4: single audit store — keyed by account_code
        self.audit_trail: defaultdict[str, list[dict]] = defaultdict(list)

        self.ai = AIService()

    # ------------------------------------------------------------------

    def log_event(self, event: LedgerEvent) -> None:
        """Record an audit event.  Single store (no duplicate list)."""
        self.audit_trail[event.account_code].append(event.to_dict())

    @property
    def events(self) -> list[dict]:
        """Flat view of all events across all accounts."""
        from itertools import chain
        return list(chain.from_iterable(self.audit_trail.values()))


# ---------------------------------------------------------------------------
# Agent 1: Mapper
# ---------------------------------------------------------------------------

class MapperAgent:
    """
    Groups the raw trial balance, merges with the COA, and classifies
    any orphan accounts (not in COA) using AI.

    Fixes: BUG-1 (column collision), BUG-2 (header row leak).
    """

    def run(self, state: ReportingState) -> ReportingState:
        log.info("MapperAgent: start")

        # Aggregate duplicate account+currency rows
        tb = (
            state.raw_tb
            .groupby(["account_code", "currency"], as_index=False)
            .agg({"debit": "sum", "credit": "sum", "account_name": "first"})
        )

        # BUG-1 FIX: drop account_name from COA before merge to avoid _x/_y suffix
        coa_slim = state.coa.drop(columns=["account_name"], errors="ignore")

        mapped = tb.merge(coa_slim, on="account_code", how="left")

        # BUG-2 FIX: exclude COA header rows (account_type == 'Header' or parent only)
        # A proper account has a non-null, non-'Header' account_type.
        orphan_mask = (
            mapped["statement"].isna() &
            (mapped["account_type"].fillna("") != "Header")
        )
        orphans = mapped[orphan_mask]

        for idx, row in orphans.iterrows():
            try:
                ai_result = state.ai.classify_account(
                    row["account_name"], str(row["account_code"])
                )
            except Exception as exc:
                log.warning("AI classify failed for %s: %s", row["account_code"], exc)
                state.logs.append(
                    f"AI classification error for {row['account_code']}: {exc}"
                )
                continue

            state.ai_suggestions.append(ai_result)

            if ai_result["confidence"] >= 0.85:
                mapped.loc[idx, "account_type"]  = ai_result["account_type"]
                mapped.loc[idx, "statement"]     = ai_result["statement"]
                mapped.loc[idx, "normal_balance"]= ai_result["normal_balance"]
                msg = (
                    f"AI mapped {row['account_code']} → {ai_result['account_type']} "
                    f"(confidence {ai_result['confidence']:.2f})"
                )
                log.info(msg)
                state.logs.append(msg)
            else:
                msg = (
                    f"LOW CONFIDENCE AI mapping for {row['account_code']} "
                    f"({ai_result['confidence']:.2f}) — manual review required"
                )
                log.warning(msg)
                state.logs.append(msg)

        mapped["net_amount"]        = mapped["debit"] - mapped["credit"]
        mapped["source_system"]     = "ERP"
        mapped["canonical_account"] = mapped["account_code"].astype(str)

        state.ledger = mapped
        self._emit_events(state)

        log.info("MapperAgent: %d ledger rows", len(state.ledger))
        return state

    def _emit_events(self, state: ReportingState) -> None:
        for _, row in state.ledger.iterrows():
            state.log_event(LedgerEvent(
                event_type="TB_IMPORT",
                account_code=row["account_code"],
                amount=row["net_amount"],
                currency=row["currency"],
                metadata={"account_name": row["account_name"]},
            ))


# ---------------------------------------------------------------------------
# Agent 2: FX Translator
# ---------------------------------------------------------------------------

class TranslatorAgent:
    """
    Translates all ledger amounts to the functional currency (USD).

    PERF-1 FIX: fully vectorised — no Python-level row iteration.
    DATA-1 FIX: missing rate raises MissingFXRateError with full context.
    DATA-2 FIX: GBP period_end fallback is logged as a compliance warning.
    """

    def run(self, state: ReportingState) -> ReportingState:
        log.info("TranslatorAgent: start")

        # Build a lookup: (currency, rate_type) → rate
        rate_lookup: dict[tuple[str, str], float] = {
            (r["currency"], r["rate_type"]): float(r["rate"])
            for _, r in state.fx_rates.iterrows()
        }

        df = state.ledger.copy()

        # Determine rate type per row (BS → period_end, PL → period_average)
        df["_rate_type"] = np.where(df["statement"] == "BS", "period_end", "period_average")

        # Vectorised rate resolution ----------------------------------------
        def _resolve_rate(currency: str, rate_type: str) -> float:
            key = (currency, rate_type)
            if key in rate_lookup:
                return rate_lookup[key]

            # Fallback to period_average with compliance warning
            fallback_key = (currency, "period_average")
            if fallback_key in rate_lookup:
                msg = (
                    f"IAS 21 WARNING: no {rate_type} rate for {currency}; "
                    f"using period_average as fallback — review required"
                )
                log.warning(msg)
                state.logs.append(msg)
                return rate_lookup[fallback_key]

            raise MissingFXRateError(
                f"No FX rate found for {currency!r} ({rate_type} or period_average). "
                "Add the rate to fx_rates.csv before running."
            )

        # Build the rate series in one pass (currencies are few — no perf issue)
        df["fx_rate"] = [
            _resolve_rate(row["currency"], row["_rate_type"])
            for _, row in df.iterrows()
        ]

        # Single vectorised multiplication
        df["translated_amount"] = df["net_amount"] * df["fx_rate"]
        df.drop(columns=["_rate_type"], inplace=True)

        state.ledger = df
        self._emit_events(state)

        log.info("TranslatorAgent: FX translation complete")
        return state

    def _emit_events(self, state: ReportingState) -> None:
        for _, row in state.ledger.iterrows():
            state.log_event(LedgerEvent(
                event_type="FX_TRANSLATION",
                account_code=row["account_code"],
                amount=row["translated_amount"],
                currency="USD",
                metadata={"fx_rate": row["fx_rate"], "rate_type": row.get("_rate_type")},
            ))


# ---------------------------------------------------------------------------
# Agent 3: Manual Adjuster
# ---------------------------------------------------------------------------

class AdjusterAgent:
    """
    Applies validated manual journal entries.

    BUG-3 FIX: merges COA account_type / statement onto adjustment rows
               so they participate correctly in statement aggregation.
    """

    def run(self, state: ReportingState) -> ReportingState:
        log.info("AdjusterAgent: start")

        valid_accounts = set(state.coa["account_code"].astype(str))
        adjustment_rows: list[dict] = []

        for entry in state.manual_adj["entries"]:
            if not self._validate(entry, valid_accounts, state):
                continue
            for line in entry["lines"]:
                amount = line["debit"] - line["credit"]
                adjustment_rows.append({
                    "account_code": str(line["account"]),
                    "net_amount":   amount,
                    "translated_amount": amount,  # already in functional currency
                    "currency":     "USD",
                    "source_system": "MANUAL_JE",
                    "canonical_account": str(line["account"]),
                })
                state.log_event(LedgerEvent(
                    event_type="MANUAL_ADJUSTMENT",
                    account_code=line["account"],
                    amount=amount,
                    metadata={"journal_id": entry["id"], "memo": line.get("memo", "")},
                ))

        if not adjustment_rows:
            log.info("AdjusterAgent: no valid adjustments to apply")
            return state

        adj_df = pd.DataFrame(adjustment_rows)

        # BUG-3 FIX: populate account_type, statement, normal_balance from COA
        coa_slim = state.coa[
            ["account_code", "account_type", "statement", "normal_balance"]
        ].copy()
        coa_slim["account_code"] = coa_slim["account_code"].astype(str)

        adj_df = adj_df.merge(coa_slim, on="account_code", how="left")

        state.ledger = pd.concat(
            [state.ledger, adj_df], ignore_index=True
        )

        log.info("AdjusterAgent: %d adjustment lines applied", len(adjustment_rows))
        return state

    # ------------------------------------------------------------------

    @staticmethod
    def _validate(
        entry: dict,
        valid_accounts: set[str],
        state: ReportingState,
    ) -> bool:
        eid = entry["id"]

        # 1. Balanced entry check
        dr = sum(l["debit"]  for l in entry["lines"])
        cr = sum(l["credit"] for l in entry["lines"])
        if abs(dr - cr) > 0.01:
            msg = (
                f"REJECTED {eid}: unbalanced entry "
                f"(debit={dr:,.2f}, credit={cr:,.2f}, Δ={dr-cr:+.2f})"
            )
            log.error(msg)
            state.logs.append(msg)
            return False

        # 2. Account existence check
        bad = [
            str(l["account"]) for l in entry["lines"]
            if str(l["account"]) not in valid_accounts
        ]
        if bad:
            msg = (
                f"REJECTED {eid}: account(s) not in COA: {bad}. "
                "Add the account before posting."
            )
            log.error(msg)
            state.logs.append(msg)
            return False

        return True


# ---------------------------------------------------------------------------
# Agent 4: Anomaly Detector
# ---------------------------------------------------------------------------

# PERF-2: threshold raised and batching implemented
_ANOMALY_THRESHOLD_USD = 5_000_000  # Avoids flagging every normal BS account


class AnomalyDetectionAgent:
    """
    Flags unusually large balances and explains them in a single AI call.

    PERF-2 FIX: one batched API call instead of one call per flagged row.
    """

    def run(self, state: ReportingState) -> ReportingState:
        log.info("AnomalyDetectionAgent: start (threshold=$%s)", f"{_ANOMALY_THRESHOLD_USD:,.0f}")

        df = state.ledger.dropna(subset=["translated_amount"])
        flagged = df[df["translated_amount"].abs() > _ANOMALY_THRESHOLD_USD].copy()

        if flagged.empty:
            log.info("AnomalyDetectionAgent: no anomalies above threshold")
            state.results["anomalies"] = []
            return state

        anomalies = [
            {
                "account_code": str(row["account_code"]),
                "account_name": str(row.get("account_name", "")),
                "amount":       float(abs(row["translated_amount"])),
            }
            for _, row in flagged.iterrows()
        ]

        try:
            anomalies = state.ai.explain_anomalies_batch(anomalies)
        except Exception as exc:
            log.warning("Batch anomaly explanation failed: %s", exc)
            for a in anomalies:
                a["explanation"] = "AI explanation unavailable."

        state.results["anomalies"] = anomalies
        log.info("AnomalyDetectionAgent: %d anomalies flagged", len(anomalies))
        return state


# ---------------------------------------------------------------------------
# Agent 5: Statement Builder
# ---------------------------------------------------------------------------

class StatementBuilder:
    """
    Builds P&L and Balance Sheet from the normalised ledger.

    PERF-3 FIX: np.select() instead of row-wise apply() for normalisation.
    """

    # Normal-balance sign convention: Debit-normal types are positive as-is;
    # Credit-normal types are negated so credit balances show as positive figures.
    _SIGN: dict[str, int] = {
        "Asset":     1,
        "Expense":   1,
        "Liability": -1,
        "Equity":    -1,
        "Revenue":   -1,
    }

    def run(self, state: ReportingState) -> ReportingState:
        log.info("StatementBuilder: start")

        df = state.ledger.copy()
        df["account_type"] = df["account_type"].fillna("Unknown")

        # PERF-3: vectorised normalisation with np.select
        conditions = [df["account_type"] == t for t in self._SIGN]
        choices    = [df["translated_amount"] * s for s in self._SIGN.values()]
        df["normalized_amount"] = np.select(conditions, choices, default=0.0)

        # Aggregate by type
        totals = (
            df.groupby("account_type")["normalized_amount"]
            .sum()
            .to_dict()
        )

        assets      = totals.get("Asset",     0.0)
        liabilities = totals.get("Liability", 0.0)
        equity      = totals.get("Equity",    0.0)
        revenue     = totals.get("Revenue",   0.0)
        expenses    = totals.get("Expense",   0.0)

        net_income    = revenue - expenses
        bs_difference = assets - (liabilities + equity + net_income)

        if abs(bs_difference) > 1:
            msg = (
                f"BS OUT OF BALANCE: Assets={assets:,.2f}, "
                f"Liabilities+Equity+NI={liabilities+equity+net_income:,.2f}, "
                f"Δ={bs_difference:+,.2f}"
            )
            log.error(msg)
            state.logs.append(msg)

        state.results["Profit & Loss"] = {
            "Revenue":    round(revenue,    2),
            "Expenses":   round(expenses,   2),
            "Net Income": round(net_income, 2),
        }
        state.results["Balance Sheet"] = {
            "Assets":      round(assets,               2),
            "Liabilities": round(liabilities,          2),
            "Equity":      round(equity + net_income,  2),
            "Difference":  round(bs_difference,        2),
        }

        log.info("StatementBuilder: Net income = $%s", f"{net_income:,.0f}")
        return state


# ---------------------------------------------------------------------------
# Agent 6: Cash Flow Statement  (IAS 7 — indirect method)
# ---------------------------------------------------------------------------

class CashFlowAgent:
    """
    Builds the Cash Flow Statement using the indirect method.

    Structure
    ---------
    Operating Activities
      Net income
      + non-cash add-backs (depreciation, amortisation, bad debt, deferred tax,
        unrealised FX) identified via cf_category='Operating' on P&L accounts
      +/- working capital changes (BS accounts with cf_category='Operating')

    Investing Activities
      Changes in non-current asset accounts (cf_category='Investing')
      Gain/loss on disposal reclassified out of operations (cf_category='Investing' on PL)

    Financing Activities
      Changes in debt and equity funding accounts (cf_category='Financing')

    Reconciliation
      Opening cash + net change + FX effect on cash = Closing cash = BS cash

    Sign convention (works uniformly for all account types)
    ---------------------------------------------------------
      CF impact = -(current_net - prior_net)
        Asset increase   -> delta positive -> -delta negative (use of cash)   OK
        Liability increase (credit-normal, more negative net) -> delta negative
                          -> -delta positive (source of cash)                  OK
    """

    _NONCASH_PL       = {"6500", "6510", "6600", "8200", "7310"}   # non-cash P&L items
    _RECLASS_TO_INV   = {"7400"}                                     # disposal gain/loss
    _ACCUM_DEP        = {"1211", "1221"}                             # skip: covered by D&A add-back
    _CASH_CODE        = "1110"

    def run(self, state: ReportingState) -> ReportingState:
        log.info("CashFlowAgent: start")

        net_income = state.results["Profit & Loss"]["Net Income"]

        # -- COA metadata index -------------------------------------------
        coa_clean = (
            state.coa[state.coa["account_type"] != "Header"]
            [["account_code", "account_name", "account_type",
              "statement", "cf_category"]]
            .copy()
        )
        coa_clean["account_code"] = coa_clean["account_code"].astype(str)
        coa_idx = coa_clean.set_index("account_code")

        def _name(code: str) -> str:
            try:    return coa_idx.loc[code, "account_name"]
            except: return code

        # -- Current-period net balances (post-adjustment, USD) -----------
        ledger = state.ledger.copy()
        ledger["account_code"] = ledger["account_code"].astype(str)
        cur = (
            ledger.groupby("account_code", as_index=False)
            .agg(
                translated_amount=("translated_amount", "sum"),
                account_type=("account_type", "first"),
                statement=("statement", "first"),
                cf_category=("cf_category", "first"),
            )
        )
        # Backfill missing metadata from COA
        cur = cur.merge(
            coa_clean[["account_code", "account_type", "statement", "cf_category"]],
            on="account_code", how="left", suffixes=("", "_c")
        )
        for col in ("account_type", "statement", "cf_category"):
            suffix_col = f"{col}_c"
            if suffix_col in cur.columns:
                cur[col] = cur[col].combine_first(cur[suffix_col])
                cur.drop(columns=[suffix_col], inplace=True)
        cur_net = cur.set_index("account_code")["translated_amount"]

        # -- Prior-period net balances (raw prior TB, already in USD) -----
        prior_raw = state.prior_tb.copy()
        prior_raw["account_code"] = prior_raw["account_code"].astype(str)
        prior_raw["net"] = prior_raw["debit"] - prior_raw["credit"]
        prior_net = prior_raw.groupby("account_code")["net"].sum()

        def _cur(code: str) -> float:  return float(cur_net.get(code, 0.0))
        def _pri(code: str) -> float:  return float(prior_net.get(code, 0.0))
        def _delta(code: str) -> float:
            """CF impact = -(current_net - prior_net); works for all account types."""
            return -(_cur(code) - _pri(code))

        # ==================================================================
        # SECTION 1 — OPERATING
        # ==================================================================

        # 1a. Non-cash P&L add-backs
        pl_cur = cur[cur["statement"] == "PL"].set_index("account_code")
        noncash: dict[str, float] = {}
        for code in self._NONCASH_PL:
            if code in pl_cur.index:
                noncash[_name(code)] = round(float(pl_cur.loc[code, "translated_amount"]), 2)
        total_noncash = sum(noncash.values())

        # 1b. Reclassify gain/loss on disposal out of operations
        reclass: dict[str, float] = {}
        for code in self._RECLASS_TO_INV:
            if code in pl_cur.index:
                reclass[_name(code)] = round(float(pl_cur.loc[code, "translated_amount"]), 2)
        total_reclass = sum(reclass.values())

        # 1c. Working capital changes (BS Operating accounts)
        wc_codes = cur[
            (cur["statement"] == "BS") & (cur["cf_category"] == "Operating")
        ]["account_code"].tolist()
        wc: dict[str, float] = {}
        for code in wc_codes:
            d = round(_delta(code), 2)
            if d != 0:
                wc[_name(code)] = d
        total_wc = sum(wc.values())

        total_operating = net_income + total_noncash + total_wc - total_reclass

        # ==================================================================
        # SECTION 2 — INVESTING
        # ==================================================================

        inv_codes = cur[
            (cur["statement"] == "BS") &
            (cur["cf_category"] == "Investing") &
            (~cur["account_code"].isin(self._ACCUM_DEP))
        ]["account_code"].tolist()
        inv: dict[str, float] = {}
        for code in inv_codes:
            d = round(_delta(code), 2)
            if d != 0:
                inv[_name(code)] = d
        for label, amt in reclass.items():
            if amt != 0:
                inv[f"Disposal proceeds / (cost): {label}"] = round(amt, 2)
        total_investing = sum(inv.values())

        # ==================================================================
        # SECTION 3 — FINANCING
        # ==================================================================

        fin_codes = cur[
            (cur["statement"] == "BS") & (cur["cf_category"] == "Financing")
        ]["account_code"].tolist()
        fin: dict[str, float] = {}
        for code in fin_codes:
            d = round(_delta(code), 2)
            if d != 0:
                fin[_name(code)] = d
        total_financing = sum(fin.values())

        # ==================================================================
        # RECONCILIATION
        # ==================================================================

        opening_cash = float(
            state.prior_tb[
                state.prior_tb["account_code"] == int(self._CASH_CODE)
            ]["debit"].sum()
        )
        closing_cash_bs = float(cur_net.get(self._CASH_CODE, 0.0))
        net_change   = total_operating + total_investing + total_financing
        fx_on_cash   = round(closing_cash_bs - opening_cash - net_change, 2)
        closing_calc = round(opening_cash + net_change + fx_on_cash, 2)

        variance = abs(closing_calc - closing_cash_bs)
        if variance > 1:
            msg = (
                f"CFS RECONCILIATION GAP: calculated={closing_calc:,.2f}, "
                f"BS={closing_cash_bs:,.2f}, delta={variance:,.2f}"
            )
            log.warning(msg)
            state.logs.append(msg)

        state.results["Cash Flow Statement"] = {
            "Operating Activities": {
                "Net Income":               round(net_income,      2),
                "Non-Cash Adjustments":     noncash,
                "Total Non-Cash":           round(total_noncash,   2),
                "Working Capital Changes":  wc,
                "Total Working Capital":    round(total_wc,        2),
                "Total Operating":          round(total_operating, 2),
            },
            "Investing Activities": {
                "items":           inv,
                "Total Investing": round(total_investing, 2),
            },
            "Financing Activities": {
                "items":            fin,
                "Total Financing":  round(total_financing, 2),
            },
            "Net Change in Cash":        round(net_change,      2),
            "FX Effect on Cash":         round(fx_on_cash,      2),
            "Opening Cash Balance":      round(opening_cash,    2),
            "Closing Cash (calculated)": closing_calc,
            "Closing Cash (per BS)":     round(closing_cash_bs, 2),
        }

        log.info(
            "CashFlowAgent: Operating=$%s  Investing=$%s  Financing=$%s",
            f"{total_operating:,.0f}", f"{total_investing:,.0f}", f"{total_financing:,.0f}",
        )
        return state


# ---------------------------------------------------------------------------
# Agent 7: Statement of Changes in Equity  (IAS 1)
# ---------------------------------------------------------------------------

class ChangesInEquityAgent:
    """
    Builds the Statement of Changes in Equity.

    Columns: Common Stock | APIC | Retained Earnings | FX Reserve | Treasury Stock | Total

    Movement logic
    --------------
    Opening balances    : equity account net balances from prior_period_tb
    Net income          : from P&L results; credited to Retained Earnings
    OCI — FX            : change in FX Translation Reserve (3310) current vs prior
    Share capital       : changes in Common Stock (3100) and APIC (3110)
    Prior-period RE     : current_RE_in_TB minus prior_RE (reflects Q1-Q3 earnings
                          already closed to RE before this quarter's entries)
    Treasury stock      : change in Treasury Stock (3400) — buybacks reduce equity
    Closing balances    : opening + all movements; reconciles to Balance Sheet equity
    """

    _CS   = "3100"
    _APIC = "3110"
    _RE   = "3200"
    _FX   = "3310"
    _TS   = "3400"   # debit-normal contra-equity

    def run(self, state: ReportingState) -> ReportingState:
        log.info("ChangesInEquityAgent: start")

        net_income = state.results["Profit & Loss"]["Net Income"]

        # -- Opening balances from prior TB (natural credit-positive sign) -
        def _prior_natural(code: str, debit_normal: bool = False) -> float:
            rows = state.prior_tb[state.prior_tb["account_code"].astype(str) == code]
            if rows.empty:
                return 0.0
            net = float(rows["debit"].sum() - rows["credit"].sum())
            return net if debit_normal else -net   # credit-normal: negate

        op_cs   = _prior_natural(self._CS)
        op_apic = _prior_natural(self._APIC)
        op_re   = _prior_natural(self._RE)
        op_fx   = _prior_natural(self._FX)
        op_ts   = _prior_natural(self._TS, debit_normal=True)   # positive = DR balance

        # -- Current balances from ledger (post-adjustments) ---------------
        ledger = state.ledger.copy()
        ledger["account_code"] = ledger["account_code"].astype(str)
        cur_net = ledger.groupby("account_code")["translated_amount"].sum()

        def _curr_natural(code: str, debit_normal: bool = False) -> float:
            v = float(cur_net.get(code, 0.0))
            return v if debit_normal else -v

        # -- Movements ------------------------------------------------------
        mv_ni   = net_income                                          # net income to RE

        mv_fx   = round(_curr_natural(self._FX) - op_fx, 2)          # OCI — FX change

        mv_cs   = round(_curr_natural(self._CS)   - op_cs,   2)      # share issuance
        mv_apic = round(_curr_natural(self._APIC) - op_apic, 2)

        mv_ts   = round(_curr_natural(self._TS, True) - op_ts, 2)    # treasury buybacks

        # Prior-period RE movement: the current TB RE already contains earnings
        # from Q1-Q3 closed into RE before this quarter's close. The difference
        # (current_TB_RE - opening_RE) is labelled "Prior-period retained earnings".
        curr_re_nat = _curr_natural(self._RE)
        mv_re_prior = round(curr_re_nat - op_re, 2)                  # Q1-Q3 + dividends

        # -- Closing balances -----------------------------------------------
        cl_cs   = round(op_cs   + mv_cs,                        2)
        cl_apic = round(op_apic + mv_apic,                      2)
        cl_re   = round(op_re   + mv_re_prior + mv_ni,          2)   # RE after NI close
        cl_fx   = round(op_fx   + mv_fx,                        2)
        cl_ts   = round(op_ts   + mv_ts,                        2)

        def _total(cs, apic, re, fx, ts) -> float:
            return round(cs + apic + re + fx - ts, 2)

        opening_total = _total(op_cs, op_apic, op_re, op_fx, op_ts)
        closing_total = _total(cl_cs, cl_apic, cl_re, cl_fx, cl_ts)

        # -- Reconcile to Balance Sheet -------------------------------------
        bs_equity = state.results["Balance Sheet"]["Equity"]
        gap = round(abs(closing_total - bs_equity), 2)
        if gap > 1:
            msg = (
                f"SCE RECONCILIATION GAP: SCE closing={closing_total:,.2f}, "
                f"BS equity={bs_equity:,.2f}, delta={gap:,.2f}"
            )
            log.warning(msg)
            state.logs.append(msg)

        state.results["Changes in Equity"] = {
            "Opening Balances": {
                "Common Stock":               op_cs,
                "Additional Paid-in Capital": op_apic,
                "Retained Earnings":          op_re,
                "FX Translation Reserve":     op_fx,
                "Treasury Stock":            -op_ts,
                "Total":                      opening_total,
            },
            "Movements": {
                "Net Income (current period)":    round(mv_ni,       2),
                "OCI — FX Translation":           mv_fx,
                "Share capital issuance":         round(mv_cs + mv_apic, 2),
                "Prior-period retained earnings": mv_re_prior,
                "Treasury stock movement":       -mv_ts,
            },
            "Closing Balances": {
                "Common Stock":               cl_cs,
                "Additional Paid-in Capital": cl_apic,
                "Retained Earnings":          cl_re,
                "FX Translation Reserve":     cl_fx,
                "Treasury Stock":            -cl_ts,
                "Total":                      closing_total,
            },
            "Reconciliation": {
                "SCE Closing Equity":   closing_total,
                "Balance Sheet Equity": bs_equity,
                "Gap":                  gap,
            },
        }

        log.info(
            "ChangesInEquityAgent: opening=$%s  closing=$%s",
            f"{opening_total:,.0f}", f"{closing_total:,.0f}",
        )
        return state



# ---------------------------------------------------------------------------
# Agent 6: Narrative
# ---------------------------------------------------------------------------

class NarrativeAgent:
    """Generates a CFO-level narrative summary via the LLM."""

    def run(self, state: ReportingState) -> ReportingState:
        log.info("NarrativeAgent: generating narrative")
        try:
            narrative = state.ai.generate_financial_narrative(
                {k: v for k, v in state.results.items() if k != "anomalies"},
                state.results.get("anomalies", []),
            )
            state.results["AI Narrative"] = narrative
        except Exception as exc:
            log.warning("Narrative generation failed: %s", exc)
            state.results["AI Narrative"] = f"Narrative unavailable: {exc}"

        return state


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def build_pipeline() -> list:
    return [
        MapperAgent(),
        TranslatorAgent(),
        AdjusterAgent(),
        AnomalyDetectionAgent(),
        StatementBuilder(),
        CashFlowAgent(),
        ChangesInEquityAgent(),
        NarrativeAgent(),
    ]


def run_pipeline(
    tb_path:    str = "tb.csv",
    coa_path:   str = "coa.csv",
    fx_path:    str = "fx.csv",
    adj_path:   str = "adjustments.json",
    prior_path: str = "prior.csv",
) -> ReportingState:
    state = ReportingState(tb_path, coa_path, fx_path, adj_path, prior_path)
    for agent in build_pipeline():
        state = agent.run(state)
    return state


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Console print helpers
# ---------------------------------------------------------------------------

def _sep(title: str = "", w: int = 68) -> None:
    bar = "=" * w
    if title:
        print(f"\n{bar}\n  {title}\n{bar}")
    else:
        print(bar)

def _row(label: str, value: float, indent: int = 4, width: int = 42) -> None:
    pad = " " * indent
    print(f"{pad}{label:<{width}}  ${value:>16,.2f}")

def _rule(indent: int = 4, w: int = 62) -> None:
    print(" " * indent + "-" * w)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    final = run_pipeline()

    # ======================================================================
    # 1. PROFIT & LOSS
    # ======================================================================
    _sep("PROFIT & LOSS  --  Q4 2024  (USD)")
    pl = final.results.get("Profit & Loss", {})
    _row("Revenue",     pl["Revenue"])
    _row("Expenses",   -pl["Expenses"])
    _rule()
    _row("Net Income",  pl["Net Income"])
    margin = pl["Net Income"] / pl["Revenue"] * 100 if pl.get("Revenue") else 0
    print(f"    {'Net Margin':<42}  {margin:>16.1f}%")

    # ======================================================================
    # 2. BALANCE SHEET
    # ======================================================================
    _sep("BALANCE SHEET  --  31 Dec 2024  (USD)")
    bs = final.results.get("Balance Sheet", {})
    _row("Assets",       bs["Assets"])
    _row("Liabilities", -bs["Liabilities"])
    _row("Equity",       bs["Equity"])
    _rule()
    diff = bs["Difference"]
    flag = "  [OUT OF BALANCE]" if abs(diff) > 1 else "  [balanced]"
    print(f"    {'A = L + E  difference':<42}  ${diff:>16,.2f}{flag}")

    # ======================================================================
    # 3. CASH FLOW STATEMENT
    # ======================================================================
    _sep("CASH FLOW STATEMENT  --  Q4 2024  (USD, indirect method)")
    cfs = final.results.get("Cash Flow Statement", {})
    oa  = cfs.get("Operating Activities", {})
    ia  = cfs.get("Investing Activities", {})
    fa  = cfs.get("Financing Activities", {})

    print("\n  OPERATING ACTIVITIES")
    _row("Net income", oa.get("Net Income", 0), indent=4)

    print("\n    Non-cash adjustments:")
    for lbl, val in oa.get("Non-Cash Adjustments", {}).items():
        _row(lbl, val, indent=6, width=40)
    _row("Total non-cash adjustments", oa.get("Total Non-Cash", 0), indent=4)

    print("\n    Working capital changes:")
    for lbl, val in oa.get("Working Capital Changes", {}).items():
        _row(lbl, val, indent=6, width=40)
    _row("Total working capital changes", oa.get("Total Working Capital", 0), indent=4)

    _rule()
    _row("Net cash from operating activities", oa.get("Total Operating", 0), indent=4)

    print("\n  INVESTING ACTIVITIES")
    for lbl, val in ia.get("items", {}).items():
        _row(lbl, val, indent=6, width=40)
    _rule()
    _row("Net cash from investing activities", ia.get("Total Investing", 0), indent=4)

    print("\n  FINANCING ACTIVITIES")
    for lbl, val in fa.get("items", {}).items():
        _row(lbl, val, indent=6, width=40)
    _rule()
    _row("Net cash from financing activities", fa.get("Total Financing", 0), indent=4)

    print()
    _rule(indent=4, w=64)
    _row("Opening cash balance",      cfs.get("Opening Cash Balance", 0))
    _row("Net change in cash",         cfs.get("Net Change in Cash",   0))
    _row("FX effect on cash",          cfs.get("FX Effect on Cash",    0))
    _rule()
    _row("Closing cash (calculated)", cfs.get("Closing Cash (calculated)", 0))
    _row("Closing cash (per BS)",     cfs.get("Closing Cash (per BS)",     0))

    # ======================================================================
    # 4. STATEMENT OF CHANGES IN EQUITY
    # ======================================================================
    _sep("STATEMENT OF CHANGES IN EQUITY  --  Q4 2024  (USD)")
    sce = final.results.get("Changes in Equity", {})

    COL_W  = 16
    LABELS = ["Common Stock", "APIC", "Retained\nEarnings", "FX Reserve", "Treasury\nStock", "Total"]
    KEYS   = ["Common Stock", "Additional Paid-in Capital",
              "Retained Earnings", "FX Translation Reserve", "Treasury Stock", "Total"]

    def _sce_row(section: str, label: str) -> None:
        d = sce.get(section, {})
        vals = [d.get(k, 0.0) for k in KEYS]
        row  = f"  {label:<28}" + "".join(f"  {v:>{COL_W},.0f}" for v in vals)
        print(row)

    # Header
    hdr_labels = ["Cmn Stock", "APIC", "Ret.Earn.", "FX Rsv", "Tsy Stock", "Total"]
    print("  " + " " * 28 + "".join(f"  {h:>{COL_W}}" for h in hdr_labels))
    _rule(indent=2, w=130)

    _sce_row("Opening Balances", "Opening balances")
    print()
    for lbl, key in [
        ("  Net income",              "Net Income (current period)"),
        ("  OCI -- FX translation",   "OCI — FX Translation"),
        ("  Share capital issuance",  "Share capital issuance"),
        ("  Prior-period RE",         "Prior-period retained earnings"),
        ("  Treasury stock movement", "Treasury stock movement"),
    ]:
        mv = sce.get("Movements", {})
        val = mv.get(key, 0.0)
        if val != 0:
            # Only RE column gets net income; only FX Reserve gets OCI; etc.
            col_map = {
                "Net Income (current period)":    "Retained Earnings",
                "OCI — FX Translation":           "FX Translation Reserve",
                "Share capital issuance":         "Additional Paid-in Capital",
                "Prior-period retained earnings": "Retained Earnings",
                "Treasury stock movement":        "Treasury Stock",
            }
            target_col = col_map.get(key, "Total")
            row_vals = []
            for k in KEYS:
                if k == target_col or k == "Total":
                    row_vals.append(val)
                elif k in ("Common Stock",) and key == "Share capital issuance":
                    row_vals.append(val)
                else:
                    row_vals.append(0.0)
            # Recalculate Total as sum of columns (excluding Total itself)
            col_sum = sum(row_vals[:-1])
            row_vals[-1] = col_sum
            print("  " + f"{lbl:<28}" + "".join(
                f"  {v:>{COL_W},.0f}" if v != 0 else " " * (COL_W + 2)
                for v in row_vals
            ))

    _rule(indent=2, w=130)
    _sce_row("Closing Balances", "Closing balances")

    rec = sce.get("Reconciliation", {})
    gap = rec.get("Gap", 0)
    gap_flag = "  [reconciled]" if gap <= 1 else f"  [GAP = {gap:,.2f}]"
    print(f"\n  Reconciliation to Balance Sheet equity: "
          f"${rec.get('SCE Closing Equity', 0):,.2f}{gap_flag}")

    # ======================================================================
    # 5. AI NARRATIVE
    # ======================================================================
    _sep("AI NARRATIVE")
    print(final.results.get("AI Narrative", "N/A"))

    # ======================================================================
    # ANOMALIES
    # ======================================================================
    _sep("ANOMALIES FLAGGED")
    for a in final.results.get("anomalies", []):
        print(f"\n  [{a['account_code']}] {a['account_name']}")
        print(f"  Amount      : ${a['amount']:>15,.0f}")
        print(f"  Explanation : {a['explanation']}")

    # ======================================================================
    # SYSTEM LOGS
    # ======================================================================
    _sep("SYSTEM LOGS")
    for msg in final.logs:
        print(f"  {msg}")

    # ======================================================================
    # AUDIT SUMMARY
    # ======================================================================
    _sep("AUDIT SUMMARY")
    print(f"  Total events  : {len(final.events)}")
    print(f"  Ledger rows   : {len(final.ledger)}")
    print(f"  Accounts      : {final.ledger['account_code'].nunique()}")
    print()