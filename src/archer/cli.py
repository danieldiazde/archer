from __future__ import annotations

import argparse
import logging
import sys

from archer.data import ingest
from archer.errors import ArcherError

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="archer",
        description="Volatility research system",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Fetch, validate, and store market data.",
    )

    ingest_parser.add_argument(
        "--config",
        default="config/data.yaml",
        help="Path to data config YAML.",
    )

    ingest_parser.add_argument(
        "--universe",
        default="config/universe.yaml",
        help="Path to universe config YAML.",
    )

    ingest_parser.add_argument(
        "--symbols",
        nargs="*",
        help="Optional subset of symbols to ingest, e.g. --symbols VXX SVXY.",
    )

    ingest_parser.add_argument(
        "--from-bronze",
        action="store_true",
        help="Rebuild silver from latest bronze instead of fetching new data.",
    )

    ingest_parser.set_defaults(func=_cmd_ingest)

    return parser


def _cmd_ingest(args: argparse.Namespace) -> int:
    reports = ingest.run_from_config(
        args.config,
        args.universe,
        symbols=args.symbols,
        from_bronze=args.from_bronze,
    )

    print(_render_report(reports))

    return 0 if all(report.status in ("ok", "flagged") for report in reports) else 1


def _render_report(reports: list[ingest.SymbolReport]) -> str:
    lines = [
        f"{'symbol':<8} {'status':<12} {'source':<12} {'rows':>8} "
        f"{'range':<24} {'gates':<24} {'silver':<8}"
    ]

    for report in reports:
        source = report.source or "—"
        date_range = _format_range(report.retrieved_start, report.retrieved_end)
        gates = ingest.summarize_gate_results(report.gate_results)
        silver = "written" if report.silver_written else "—"

        lines.append(
            f"{report.symbol:<8} {report.status:<12} {source:<12} "
            f"{report.rows:>8} {date_range:<24} {gates:<24} {silver:<8}"
        )

        if report.error:
            lines.append(f"  error: {report.error}")
        
        for result in report.gate_results:
            if result.status in ("failed", "flagged"):
                detail = f": {result.detail}" if result.detail else ""
                rows = ""

                if result.bad_rows is not None:
                    rows = f" | bad_rows={len(result.bad_rows)}"

                lines.append(
                    f"  {result.status}: {result.gate}{detail}{rows}"
                )

    return "\n".join(lines)


def _format_range(start: str | None, end: str | None) -> str:
    if start is None or end is None:
        return "—"

    return f"{start}..{end}"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s · %(message)s",
        stream=sys.stderr,
    )

    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except ArcherError as exc:
        logger.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())