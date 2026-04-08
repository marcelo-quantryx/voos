#!/usr/bin/env python3
"""
find_cheapest_flight.py
────────────────────────────────────────────────────────────────
Find the cheapest day to fly between two airports using the
Amadeus for Developers API (free tier).

Sign up for free at: https://developers.amadeus.com/
You'll get a client_id and client_secret for the test environment.

Usage:
    python find_cheapest_flight.py \
        --origin MAD \
        --destination LHR \
        --start-date 2026-06-01 \
        --end-date 2026-06-30

    python find_cheapest_flight.py --deploy

Dependencies:
    pip install requests python-dateutil rich prefect

NOTE on parallelism:
    These are network-bound requests, so we use ThreadPoolExecutor rather
    than multiprocessing.Pool. Threads share memory (ideal for the shared
    progress bar + result list) and avoid the serialisation overhead of
    spawning separate processes for pure I/O work. Each worker holds its
    own HTTP session and refreshes its token independently.
"""

import argparse
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, timedelta, datetime
from typing import Optional

from prefect import flow, task
from prefect.variables import Variable

import duckdb
import requests
from dateutil.parser import parse as parse_date
from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel
from rich.text import Text
from rich.progress import (
    Progress, SpinnerColumn, BarColumn,
    TextColumn, TimeElapsedColumn, MofNCompleteColumn,
)

# ── Constants ──────────────────────────────────────────────────────────────────
AMADEUS_AUTH_URL   = "https://test.api.amadeus.com/v1/security/oauth2/token"
AMADEUS_OFFERS_URL = "https://test.api.amadeus.com/v2/shopping/flight-offers"
AIRLINES_URL       = "https://test.api.amadeus.com/v1/reference-data/airlines"

console = Console()

_airline_cache = {}
_airline_cache_lock = threading.Lock()


# ── Thread-local token management ─────────────────────────────────────────────
@dataclass
class WorkerState:
    """Each worker thread keeps its own session and token."""
    client_id:     str
    client_secret: str
    token:         str = field(default="")
    token_time:    float = field(default=0.0)
    session:       requests.Session = field(default_factory=requests.Session)
    lock:          threading.Lock = field(default_factory=threading.Lock)

    def ensure_token(self) -> str:
        """Return a valid token, refreshing if it is about to expire (>25 min old)."""
        with self.lock:
            if not self.token or time.time() - self.token_time > 1500:
                resp = self.session.post(
                    AMADEUS_AUTH_URL,
                    data={
                        "grant_type":    "client_credentials",
                        "client_id":     self.client_id,
                        "client_secret": self.client_secret,
                    },
                    timeout=10,
                )
                resp.raise_for_status()
                self.token      = resp.json()["access_token"]
                self.token_time = time.time()
            return self.token


# Thread-local storage: one WorkerState per thread
_thread_local = threading.local()


def get_worker(client_id: str, client_secret: str) -> WorkerState:
    """Lazily create a WorkerState for the current thread."""
    if not hasattr(_thread_local, "state"):
        _thread_local.state = WorkerState(
            client_id=client_id,
            client_secret=client_secret,
        )
    return _thread_local.state

def get_airline_name(worker: WorkerState, code: str) -> str:
    if not code or code == "N/A":
        return "N/A"
    with _airline_cache_lock:
        if code in _airline_cache:
            return _airline_cache[code]
    
    token = worker.ensure_token()
    try:
        resp = worker.session.get(
            AIRLINES_URL,
            headers={"Authorization": f"Bearer {token}"},
            params={"airlineCodes": code},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json().get("data", [])
            name = data[0]["businessName"] if data else code
        else:
            name = code
    except requests.RequestException:
        name = code
        
    with _airline_cache_lock:
        _airline_cache[code] = name
        
    return name

# ── Single-date search (runs inside a worker thread) ──────────────────────────
def fetch_date(
    dep_date:     str,
    client_id:    str,
    client_secret: str,
    origin:       str,
    destination:  str,
    adults:       int,
    currency:     str,
    non_stop:     bool,
) -> Optional[dict]:
    """
    Called concurrently by the thread pool.
    Returns the cheapest offer for dep_date, or None if unavailable.
    """
    worker = get_worker(client_id, client_secret)

    for attempt in range(2):          # retry once on 401
        token = worker.ensure_token()
        params = {
            "originLocationCode":      origin.upper(),
            "destinationLocationCode": destination.upper(),
            "departureDate":           dep_date,
            "adults":                  adults,
            "max":                     5,
            "currencyCode":            currency,
            "nonStop":                 str(non_stop).lower(),
        }
        try:
            resp = worker.session.get(
                AMADEUS_OFFERS_URL,
                headers={"Authorization": f"Bearer {token}"},
                params=params,
                timeout=15,
            )
        except requests.RequestException:
            return None

        if resp.status_code == 401 and attempt == 0:
            # Force token refresh on next loop
            with worker.lock:
                worker.token = ""
            continue

        if resp.status_code != 200:
            return None

        offers = resp.json().get("data", [])
        if not offers:
            return None

        cheapest  = min(offers, key=lambda o: float(o["price"]["grandTotal"]))
        itinerary = cheapest["itineraries"][0]
        segments  = itinerary["segments"]

        airline_code = cheapest.get("validatingAirlineCodes", ["N/A"])[0]
        airline_name = get_airline_name(worker, airline_code)
        
        flight_nums = []
        for seg in segments:
            c = seg.get("carrierCode", "")
            n = seg.get("number", "")
            flight_nums.append(f"{c}{n}")
        flight_number = ", ".join(flight_nums)

        return {
            "date":     dep_date,
            "price":    float(cheapest["price"]["grandTotal"]),
            "currency": cheapest["price"]["currency"],
            "airline_code": airline_code,
            "airline_name": airline_name,
            "flight_number": flight_number,
            "stops":    len(segments) - 1,
            "duration": itinerary.get("duration", "N/A"),
            "dep_time": segments[0]["departure"]["at"][11:16],
            "arr_time": segments[-1]["arrival"]["at"][11:16],
        }

    return None


# ── Date range generator ───────────────────────────────────────────────────────
def date_range(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


# ── Display ────────────────────────────────────────────────────────────────────
def display_results(results: list[dict], origin: str, destination: str, top_n: int = 10):
    if not results:
        console.print("[yellow]No results found for the given route/dates.[/yellow]")
        return

    sorted_r       = sorted(results, key=lambda x: x["price"])
    cheapest_price = sorted_r[0]["price"]
    priciest       = sorted_r[-1]["price"]
    price_range    = priciest - cheapest_price + 0.01

    console.print(Panel(
        Text(f"✈  {origin.upper()}  →  {destination.upper()}  ✈", style="bold cyan"),
        expand=False,
    ))

    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
        title=(
            f"Top {min(top_n, len(sorted_r))} Cheapest Days  "
            f"(out of {len(sorted_r)} dates searched)"
        ),
        title_style="bold white",
        padding=(0, 1),
    )
    table.add_column("#",        style="dim", width=4,  justify="right")
    table.add_column("Date",                  width=13)
    table.add_column("Weekday",  style="yellow", width=11)
    table.add_column("Price",                 width=14, justify="right")
    table.add_column("Flight No.",            width=14, justify="center")
    table.add_column("Airline",               width=18, justify="center")
    table.add_column("Stops",                 width=8,  justify="center")
    table.add_column("Duration",              width=11)
    table.add_column("Dep",                   width=7,  justify="center")
    table.add_column("Arr",                   width=7,  justify="center")

    for i, r in enumerate(sorted_r[:top_n], 1):
        d         = parse_date(r["date"])
        price_str = f"{r['currency']} {r['price']:.2f}"
        stops_str = "direct" if r["stops"] == 0 else f"{r['stops']} stop{'s' if r['stops'] > 1 else ''}"
        duration  = r.get("duration", "—").replace("PT", "").lower()

        pct = (r["price"] - cheapest_price) / price_range
        if pct < 0.2:
            price_col = f"[bold green]{price_str}[/bold green]"
        elif pct < 0.5:
            price_col = f"[bold yellow]{price_str}[/bold yellow]"
        else:
            price_col = f"[bold red]{price_str}[/bold red]"

        table.add_row(
            str(i),
            f"[cyan]{r['date']}[/cyan]",
            d.strftime("%A"),
            price_col,
            r.get("flight_number", "—"),
            r.get("airline_name", r.get("airline_code", "—")),
            stops_str,
            duration,
            r.get("dep_time", "—"),
            r.get("arr_time", "—"),
            style="on dark_green" if i == 1 else "",
        )

    console.print(table)

    best    = sorted_r[0]
    savings = priciest - cheapest_price
    console.print(
        f"\n💡 [bold green]Best day:[/bold green] "
        f"[cyan]{best['date']}[/cyan] "
        f"([yellow]{parse_date(best['date']).strftime('%A')}[/yellow])  "
        f"[bold green]{best['currency']} {best['price']:.2f}[/bold green]"
    )
    if savings > 0.01:
        console.print(
            f"   Saves up to [bold]{best['currency']} {savings:.2f}[/bold] "
            f"vs the priciest date in range."
        )

    # Day-of-week averages
    dow: dict[str, list[float]] = {}
    for r in results:
        dow.setdefault(parse_date(r["date"]).strftime("%A"), []).append(r["price"])
    if len(dow) > 1:
        avg      = {d: sum(p) / len(p) for d, p in dow.items()}
        best_dow  = min(avg, key=avg.__getitem__)
        worst_dow = max(avg, key=avg.__getitem__)
        curr      = results[0]["currency"]
        console.print(
            f"\n📅 [bold]Day-of-week:[/bold] "
            f"cheapest → [green]{best_dow}s[/green] ({curr} {avg[best_dow]:.0f} avg)  "
            f"· priciest → [red]{worst_dow}s[/red] ({curr} {avg[worst_dow]:.0f} avg)"
        )


# ── Prefect Flow ───────────────────────────────────────────────────────────────
@flow(name="find-cheapest-flight")
def find_cheapest_flight(
    origin: str,
    destination: str,
    start_date: str,
    end_date: str,
    adults: int = 1,
    currency: str = "EUR",
    non_stop: bool = False,
    top: int = 10,
    workers: int = 5,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    export_duckdb: Optional[str] = None,
):
    """
    Prefect flow to find the cheapest flights over a date range.
    """
    # Fetch from args, env, or Prefect Variable
    resolved_client_id     = client_id     or os.environ.get("AMADEUS_CLIENT_ID")     or Variable.get("amadeus_client_id")
    resolved_client_secret = client_secret or os.environ.get("AMADEUS_CLIENT_SECRET") or Variable.get("amadeus_client_secret")

    if not resolved_client_id or not resolved_client_secret:
        msg = (
            "Amadeus API credentials missing! Set workspace variables "
            "AMADEUS_CLIENT_ID and AMADEUS_CLIENT_SECRET, or pass them as args."
        )
        console.print(f"[bold red]Error:[/bold red] {msg}")
        raise ValueError(msg)

    try:
        start = parse_date(start_date).date()
        end   = parse_date(end_date).date()
    except ValueError as e:
        console.print(f"[red]Invalid date:[/red] {e}")
        raise ValueError(f"Invalid date: {e}")

    if start > end:
        console.print("[red]start_date must be before end_date[/red]")
        raise ValueError("start_date must be before end_date")
    if start < date.today():
        console.print("[yellow]⚠  start_date is in the past — Amadeus may reject past dates.[/yellow]")

    days = [d.strftime("%Y-%m-%d") for d in date_range(start, end)]

    console.print(
        f"\n[bold]Route:[/bold]   {origin.upper()} → {destination.upper()}\n"
        f"[bold]Range:[/bold]   {start} → {end}  ({len(days)} days)\n"
        f"[bold]Workers:[/bold] {workers} parallel threads  |  "
        f"[bold]Adults:[/bold] {adults}  |  "
        f"[bold]Currency:[/bold] {currency}  |  "
        f"[bold]Non-stop:[/bold] {'yes' if non_stop else 'no'}\n"
    )

    # ── Warm up: verify credentials before spawning threads ──────────────────
    console.print("[bold]Authenticating...[/bold]")
    try:
        warm = WorkerState(client_id=resolved_client_id, client_secret=resolved_client_secret)
        warm.ensure_token()
        console.print("[green]✓ Credentials valid[/green]\n")
    except requests.HTTPError as e:
        console.print(f"[red]Authentication failed:[/red] {e}")
        raise ValueError(f"Authentication failed: {e}")

    # ── Parallel search ───────────────────────────────────────────────────────
    results: list[dict] = []
    errors   = 0
    lock     = threading.Lock()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_prog = progress.add_task(
            f"Searching with [cyan]{workers}[/cyan] workers...",
            total=len(days),
        )

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    fetch_date,
                    dep_date     = dep,
                    client_id    = resolved_client_id,
                    client_secret= resolved_client_secret,
                    origin       = origin,
                    destination  = destination,
                    adults       = adults,
                    currency     = currency,
                    non_stop     = non_stop,
                ): dep
                for dep in days
            }

            for future in as_completed(futures):
                dep = futures[future]
                try:
                    result = future.result()
                    with lock:
                        if result:
                            results.append(result)
                        else:
                            errors += 1
                except Exception as exc:
                    with lock:
                        errors += 1
                    console.print(f"[yellow]  ⚠ {dep}: {exc}[/yellow]")
                finally:
                    progress.advance(task_prog)

    console.print()
    if errors:
        console.print(
            f"[yellow]{errors} date(s) returned no results "
            f"(no flights on that day, or API error).[/yellow]\n"
        )

    display_results(results, origin, destination, top_n=top)
    console.print()

    if results and export_duckdb:
        sorted_r = sorted(results, key=lambda x: x["price"])
        best = sorted_r[0]
        
        try:
            conn = duckdb.connect(export_duckdb)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS flight_prices (
                    recorded_at TIMESTAMP,
                    origin VARCHAR,
                    destination VARCHAR,
                    flight_date DATE,
                    flight_number VARCHAR,
                    airline_name VARCHAR,
                    price DOUBLE,
                    currency VARCHAR
                )
            """)
            conn.execute("""
                INSERT INTO flight_prices VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(),
                origin,
                destination,
                parse_date(best["date"]).date(),
                best.get("flight_number", ""),
                best.get("airline_name", best.get("airline_code", "")),
                best["price"],
                best["currency"]
            ))
            conn.close()
            console.print(f"[green]✓ Best price exported to DuckDB: {export_duckdb}[/green]")
        except Exception as e:
            console.print(f"[red]Failed to export to DuckDB: {e}[/red]")


# ── Main / CLI ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find the cheapest day to fly (parallel) using the Amadeus API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python find_cheapest_flight.py --origin MAD --destination LHR \\
      --start-date 2026-06-01 --end-date 2026-06-30

  python find_cheapest_flight.py --deploy

Environment variables:
  AMADEUS_CLIENT_ID
  AMADEUS_CLIENT_SECRET
        """,
    )
    parser.add_argument("--origin",        required=False, help="Origin airport code")
    parser.add_argument("--destination",   required=False, help="Destination airport code")
    parser.add_argument("--start-date",    required=False, help="YYYY-MM-DD")
    parser.add_argument("--end-date",      required=False, help="YYYY-MM-DD")
    parser.add_argument("--adults",        type=int,   default=1,    help="Passengers (default: 1)")
    parser.add_argument("--currency",                  default="EUR")
    parser.add_argument("--non-stop",      action="store_true")
    parser.add_argument("--top",           type=int,   default=10,   help="Rows to display (default: 10)")
    parser.add_argument("--workers",       type=int,   default=5,
                        help="Parallel threads (default: 5). Stay ≤10 on the free tier to avoid rate-limit errors.")
    parser.add_argument("--client-id",     help="Amadeus client ID")
    parser.add_argument("--client-secret", help="Amadeus client secret")
    parser.add_argument("--export-duckdb", help="Path to DuckDB file to record the best flight")
    parser.add_argument("--deploy",        action="store_true", help="Deploy the flow to Prefect Cloud using GitHub source")

    args = parser.parse_args()

    if args.deploy:
        console.print("[bold cyan]Deploying to Prefect Cloud...[/bold cyan]")
        find_cheapest_flight.from_source(
            source="https://github.com/marcelo-quantryx/voos.git",
            entrypoint="find_cheapest_flight.py:find_cheapest_flight"
        ).deploy(
            name="amadeus-flight-search",
            work_pool_name="default",
            interval=timedelta(hours=1),
        )
        console.print("[bold green]Deployment successful![/bold green]")
    else:
        if not all([args.origin, args.destination, args.start_date, args.end_date]):
            console.print("[red]Error: --origin, --destination, --start-date, and --end-date are required when not deploying.[/red]")
            sys.exit(1)
            
        find_cheapest_flight(
            origin=args.origin,
            destination=args.destination,
            start_date=args.start_date,
            end_date=args.end_date,
            adults=args.adults,
            currency=args.currency,
            non_stop=args.non_stop,
            top=args.top,
            workers=args.workers,
            client_id=args.client_id,
            client_secret=args.client_secret,
            export_duckdb=args.export_duckdb,
        )