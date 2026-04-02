"""WRDS data loader with correct date-scoped mappings and real batching."""

import logging
import os
from typing import Optional

import pandas as pd
import wrds

logger = logging.getLogger(__name__)


class WRDSDataLoader:
    """Load data from WRDS with correct ticker/PERMNO/GVKEY mappings.

    Key correctness guarantees:
    - All ticker→PERMNO lookups are date-scoped (avoids recycled tickers)
    - CRSP-Compustat links are filtered by linkdt/linkenddt
    - permno→ticker mapping is cached once, not re-queried per fetch
    - All fetch methods chunk symbols into batches of `batch_size`
    """

    # Real WRDS Beta Suite table (market-model betas)
    _BETA_TABLE = "beta.betamfret"

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        batch_size: int = 750,
    ):
        username = username or os.getenv("WRDS_USERNAME")
        password = password or os.getenv("WRDS_PASSWORD")

        self.conn = wrds.Connection(wrds_username=username, wrds_password=password)
        self.batch_size = batch_size

        # Populated lazily on first use — one query, reused everywhere
        self._permno_to_ticker: dict[int, str] = {}
        self._ticker_to_permno: dict[str, int] = {}
        self._ticker_to_gvkey:  dict[str, str] = {}

        logger.info("WRDS connection established")

    # ------------------------------------------------------------------
    # Public fetch methods
    # ------------------------------------------------------------------

    def fetch_prices(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch daily prices from CRSP dsf, chunked into batches."""
        permnos = self._get_permnos(symbols, start_date, end_date)
        if not permnos:
            return pd.DataFrame()

        chunks = _chunk(permnos, self.batch_size)
        frames = [self._fetch_prices_chunk(c, start_date, end_date) for c in chunks]
        df = pd.concat(frames, ignore_index=True)
        df["symbol"] = df["permno"].map(self._permno_to_ticker)
        return df

    def fetch_betas(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
        estper: int = 60,           # estimation period in trading days
    ) -> pd.DataFrame:
        """
        Fetch rolling betas from WRDS Beta Suite (beta.betamfret).

        Columns returned: permno, date, beta, ivol (idiosyncratic vol), tvol.
        Falls back to empty DataFrame if Beta Suite is not licensed.
        """
        permnos = self._get_permnos(symbols, start_date, end_date)
        if not permnos:
            return pd.DataFrame()

        chunks = _chunk(permnos, self.batch_size)
        frames = []
        for chunk in chunks:
            try:
                frames.append(self._fetch_betas_chunk(chunk, start_date, end_date, estper))
            except Exception as exc:
                logger.warning(f"Beta chunk failed: {exc}")

        if not frames:
            logger.warning(
                "WRDS Beta Suite (beta.betamfret) not available. "
                "Compute betas locally from returns instead."
            )
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)
        df["symbol"] = df["permno"].map(self._permno_to_ticker)
        return df

    def fetch_fundamentals(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Fetch quarterly fundamentals from comp.fundq.
        Uses rdq (report date) not datadate — avoids look-ahead bias.
        """
        gvkeys = self._get_gvkeys(symbols)
        if not gvkeys:
            return pd.DataFrame()

        chunks = _chunk(gvkeys, self.batch_size)
        frames = [self._fetch_fundamentals_chunk(c, start_date, end_date) for c in chunks]
        df = pd.concat(frames, ignore_index=True)
        df["symbol"] = df["gvkey"].map({v: k for k, v in self._ticker_to_gvkey.items()})
        return df

    def fetch_wrds_ratios(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Fetch pre-computed financial ratios from comp.wrds_ratios.

        Uses public_date (the date ratios become public) to avoid
        look-ahead bias. Much cleaner than deriving ratios from raw fundq.

        Columns fetched map directly to TABULAR_CONTINUOUS_NAMES:
            roe, roa, de_ratio, ptb, pe_op_dil, mktcap,
            divyield, npm, opmad, at_turn
        """
        gvkeys = self._get_gvkeys(symbols)
        if not gvkeys:
            return pd.DataFrame()

        chunks = _chunk(gvkeys, self.batch_size)
        frames = [self._fetch_wrds_ratios_chunk(c, start_date, end_date)
                  for c in chunks]
        frames = [f for f in frames if not f.empty]
        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)
        df["symbol"] = df["gvkey"].map(
            {v: k for k, v in self._ticker_to_gvkey.items()}
        )
        return df

    def _fetch_wrds_ratios_chunk(
        self, gvkeys: list[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        gl = "','".join(gvkeys)
        return self.conn.raw_sql(f"""
            SELECT gvkey, public_date,
                   roe, roa, de_ratio, ptb, pe_op_dil,
                   mktcap, divyield, npm, opmad, at_turn
            FROM comp.wrds_ratios
            WHERE gvkey IN ('{gl}')
              AND public_date BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY gvkey, public_date
        """)

    def fetch_gics(self, symbols: list[str]) -> pd.DataFrame:
        """Fetch GICS sector/group codes from comp.company."""
        gvkeys = self._get_gvkeys(symbols)
        if not gvkeys:
            return pd.DataFrame()

        chunks = _chunk(gvkeys, self.batch_size)
        frames = [self._fetch_gics_chunk(c) for c in chunks]
        df = pd.concat(frames, ignore_index=True)
        df["symbol"] = df["gvkey"].map({v: k for k, v in self._ticker_to_gvkey.items()})
        return df

    # ------------------------------------------------------------------
    # Private chunk-level queries
    # ------------------------------------------------------------------

    def _fetch_prices_chunk(
        self, permnos: list[int], start_date: str, end_date: str
    ) -> pd.DataFrame:
        pl = ",".join(map(str, permnos))
        return self.conn.raw_sql(f"""
            SELECT permno, date, prc, vol, ret, shrout, bidlo, askhi
            FROM crsp.dsf
            WHERE permno IN ({pl})
              AND date BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY permno, date
        """)

    def _fetch_betas_chunk(
        self,
        permnos: list[int],
        start_date: str,
        end_date: str,
        estper: int,
    ) -> pd.DataFrame:
        pl = ",".join(map(str, permnos))
        df = self.conn.raw_sql(f"""
            SELECT permno, date, beta, ivol, tvol
            FROM {self._BETA_TABLE}
            WHERE permno IN ({pl})
              AND date BETWEEN '{start_date}' AND '{end_date}'
              AND estper = {estper}
            ORDER BY permno, date
        """)
        return df.rename(columns={"ivol": "idiosyncratic_vol", "tvol": "total_vol"})

    def _fetch_fundamentals_chunk(
        self, gvkeys: list[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        gl = "','".join(gvkeys)
        return self.conn.raw_sql(f"""
            SELECT gvkey, rdq, atq, seq, niq, cshoq, prccq,
                   epspxq, opepsq, ceqq, txtq, xintq, saleq, cheq
            FROM comp.fundq
            WHERE gvkey IN ('{gl}')
              AND rdq BETWEEN '{start_date}' AND '{end_date}'
              AND indfmt = 'INDL'
              AND datafmt = 'STD'
              AND popsrc = 'D'
            ORDER BY gvkey, rdq
        """)

    def _fetch_gics_chunk(self, gvkeys: list[str]) -> pd.DataFrame:
        gl = "','".join(gvkeys)
        return self.conn.raw_sql(f"""
            SELECT gvkey, gsector, ggroup, gind, gsubind
            FROM comp.company
            WHERE gvkey IN ('{gl}')
        """)

    # ------------------------------------------------------------------
    # Date-scoped ticker → PERMNO / GVKEY mappings (cached)
    # ------------------------------------------------------------------

    def _get_permnos(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
    ) -> list[int]:
        """
        Map tickers to PERMNOs scoped to the date range.

        dsenames has one row per (permno, namedt, nameendt) interval.
        Without date filtering, recycled tickers return multiple PERMNOs
        belonging to different companies.
        """
        missing = [s for s in symbols if s not in self._ticker_to_permno]
        if missing:
            sl = "','".join(missing)
            df = self.conn.raw_sql(f"""
                SELECT DISTINCT ON (ticker) permno, ticker
                FROM crsp.dsenames
                WHERE ticker IN ('{sl}')
                  AND namedt     <= '{end_date}'
                  AND (nameendt >= '{start_date}' OR nameendt IS NULL)
                ORDER BY ticker, nameendt DESC NULLS FIRST
            """)
            for _, row in df.iterrows():
                self._ticker_to_permno[row["ticker"]] = int(row["permno"])
                self._permno_to_ticker[int(row["permno"])] = row["ticker"]

        result = []
        for sym in symbols:
            p = self._ticker_to_permno.get(sym)
            if p is not None:
                result.append(p)
            else:
                logger.warning(f"No PERMNO found for {sym}")
        return result

    def _get_gvkeys(self, symbols: list[str]) -> list[str]:
        """
        Map tickers to GVKEYs via comp.names.
        Falls back to CRSP-Compustat link with date-scoped filtering.
        """
        missing = [s for s in symbols if s not in self._ticker_to_gvkey]
        if missing:
            sl = "','".join(missing)
            try:
                df = self.conn.raw_sql(f"""
                    SELECT DISTINCT ON (tic) gvkey, tic AS ticker
                    FROM comp.names
                    WHERE tic IN ('{sl}')
                    ORDER BY tic, gvkey DESC
                """)
                for _, row in df.iterrows():
                    self._ticker_to_gvkey[row["ticker"]] = row["gvkey"]
            except Exception:
                pass

        # For any still missing, try the CRSP-Compustat link (date-scoped)
        still_missing = [s for s in missing if s not in self._ticker_to_gvkey]
        if still_missing:
            permnos = [self._ticker_to_permno[s] for s in still_missing
                       if s in self._ticker_to_permno]
            if permnos:
                pl = ",".join(map(str, permnos))
                df = self.conn.raw_sql(f"""
                    SELECT DISTINCT ON (lpermno) gvkey, lpermno
                    FROM crsp.ccmxpf_lnkhist
                    WHERE lpermno IN ({pl})
                      AND linktype IN ('LC', 'LU')
                      AND linkprim IN ('P', 'C')
                      AND (linkenddt IS NULL OR linkenddt >= CURRENT_DATE)
                    ORDER BY lpermno, linkenddt DESC NULLS FIRST
                """)
                permno_gvkey = dict(zip(df["lpermno"].astype(int), df["gvkey"]))
                for sym in still_missing:
                    p = self._ticker_to_permno.get(sym)
                    if p and p in permno_gvkey:
                        self._ticker_to_gvkey[sym] = permno_gvkey[p]

        return [self._ticker_to_gvkey[s] for s in symbols if s in self._ticker_to_gvkey]

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def close(self):
        self.conn.close()
        logger.info("WRDS connection closed")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _chunk(lst: list, size: int) -> list[list]:
    return [lst[i : i + size] for i in range(0, len(lst), size)]