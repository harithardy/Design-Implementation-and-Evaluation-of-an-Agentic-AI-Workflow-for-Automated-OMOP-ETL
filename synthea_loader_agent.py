from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SYNTHEA_DIR = PROJECT_ROOT / "data" / "synthea_tables"
DEFAULT_DATABASE: str | None = None
DEFAULT_SCHEMA = "synthea"
DEFAULT_BATCH_SIZE = 5_000
DEFAULT_SAMPLE_ROWS = 250

UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
DATETIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?$")


@dataclass(frozen=True)
class ColumnPlan:
    name: str
    sql_type: str
    value_kind: Literal["text", "date", "datetime", "uuid"]


@dataclass(frozen=True)
class TablePlan:
    source_file: Path
    table_name: str
    delimiter: str
    columns: tuple[ColumnPlan, ...]


@dataclass(frozen=True)
class LoadResult:
    source_file: str
    table_name: str
    status: Literal["loaded"]
    rows_loaded: int
    reason: str


class SyntheaLoaderState(TypedDict, total=False):
    synthea_directory: Path
    target_database: str | None
    schema: str
    table_prefix: str
    batch_size: int
    sample_rows: int
    execute: bool
    max_rows_per_file: int | None
    engine: Any
    synthea_files: list[Path]
    table_plans: list[TablePlan]
    execution_results: list[LoadResult]
    summary: dict[str, Any]


def quote_identifier(identifier: str) -> str:
    return f"[{identifier.replace(']', ']]')}]"


def detect_delimiter(file_path: Path) -> str:
    with file_path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t|")
        return dialect.delimiter
    except csv.Error:
        return ","


def is_uuid_value(value: str) -> bool:
    return bool(UUID_RE.match(value))


def is_date_value(value: str) -> bool:
    return bool(DATE_RE.match(value))


def is_datetime_value(value: str) -> bool:
    return bool(DATETIME_RE.match(value))


def infer_column_plan(column_name: str, sample_values: list[str]) -> ColumnPlan:
    non_empty_values = [value.strip() for value in sample_values if value.strip()]
    if non_empty_values and all(is_uuid_value(value) for value in non_empty_values):
        return ColumnPlan(name=column_name, sql_type="UNIQUEIDENTIFIER", value_kind="uuid")
    if non_empty_values and all(is_datetime_value(value) for value in non_empty_values):
        return ColumnPlan(name=column_name, sql_type="DATETIME2", value_kind="datetime")
    if non_empty_values and all(is_date_value(value) for value in non_empty_values):
        return ColumnPlan(name=column_name, sql_type="DATE", value_kind="date")
    return ColumnPlan(name=column_name, sql_type="NVARCHAR(MAX)", value_kind="text")


def sample_file_columns(
    file_path: Path,
    delimiter: str,
    sample_rows: int,
) -> tuple[list[str], list[list[str]]]:
    with file_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        header = next(reader)
        columns = [[] for _ in header]

        for row_number, row in enumerate(reader, start=1):
            if len(row) != len(header):
                raise ValueError(
                    f"{file_path.name} row {row_number} has {len(row)} columns, "
                    f"expected {len(header)}."
                )
            for index, value in enumerate(row):
                columns[index].append(value)
            if row_number >= sample_rows:
                break

    if len(set(header)) != len(header):
        raise ValueError(f"{file_path.name} contains duplicate header names: {header}")
    return header, columns


def build_create_table_statement(table_plan: TablePlan, schema: str) -> str:
    column_definitions = ", ".join(
        f"{quote_identifier(column.name)} {column.sql_type} NULL"
        for column in table_plan.columns
    )
    return (
        f"CREATE TABLE {quote_identifier(schema)}.{quote_identifier(table_plan.table_name)} "
        f"({column_definitions})"
    )


def build_target_table_name(source_file: Path, table_prefix: str = "") -> str:
    return f"{table_prefix}{source_file.stem}"


def resolve_target_database(engine: Any, requested_database: str | None) -> str | None:
    if requested_database:
        return requested_database

    engine_url = getattr(engine, "url", None)
    database_name = getattr(engine_url, "database", None)
    return str(database_name) if database_name else None


def build_use_database_statement(target_database: str) -> str:
    return f"USE {quote_identifier(target_database)}"


def ensure_schema_exists(connection: Any, schema: str) -> None:
    connection.execute(
        text(
            "IF SCHEMA_ID(:schema_name) IS NULL "
            f"EXEC(N'CREATE SCHEMA {quote_identifier(schema)}')"
        ),
        {"schema_name": schema},
    )


def get_engine() -> Any:
    try:
        from src.connection import engine
    except ModuleNotFoundError:
        from connection import engine

    return engine


def discover_synthea_files(state: SyntheaLoaderState) -> SyntheaLoaderState:
    synthea_directory = Path(state.get("synthea_directory", DEFAULT_SYNTHEA_DIR))
    synthea_files = sorted(synthea_directory.glob("*.csv"), key=lambda path: path.name.lower())
    if not synthea_files:
        raise FileNotFoundError(f"No CSV files were found in {synthea_directory}.")
    return {"synthea_files": synthea_files}


def infer_table_plans(state: SyntheaLoaderState) -> SyntheaLoaderState:
    sample_rows = int(state.get("sample_rows", DEFAULT_SAMPLE_ROWS))
    table_prefix = str(state.get("table_prefix", ""))
    table_plans: list[TablePlan] = []

    for file_path in state["synthea_files"]:
        delimiter = detect_delimiter(file_path)
        header, column_samples = sample_file_columns(file_path, delimiter, sample_rows)
        table_plans.append(
            TablePlan(
                source_file=file_path,
                table_name=build_target_table_name(file_path, table_prefix),
                delimiter=delimiter,
                columns=tuple(
                    infer_column_plan(column_name, values)
                    for column_name, values in zip(header, column_samples, strict=True)
                ),
            )
        )

    return {"table_plans": table_plans}


def should_execute_plan(state: SyntheaLoaderState) -> str:
    return "create" if state.get("execute", True) else "summarize"


def create_target_tables(state: SyntheaLoaderState) -> SyntheaLoaderState:
    engine = state.get("engine") or get_engine()
    target_database = resolve_target_database(engine, state.get("target_database", DEFAULT_DATABASE))
    schema = state.get("schema", DEFAULT_SCHEMA)

    with engine.connect() as connection:
        if target_database:
            connection.execute(text(build_use_database_statement(target_database)))
        ensure_schema_exists(connection, schema)
        for table_plan in state["table_plans"]:
            connection.execute(
                text(
                    f"DROP TABLE IF EXISTS {quote_identifier(schema)}."
                    f"{quote_identifier(table_plan.table_name)}"
                )
            )
            connection.execute(text(build_create_table_statement(table_plan, schema)))

    return {}


def parse_datetime_value(value: str) -> datetime:
    if value.endswith("Z"):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is not None:
        return parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def coerce_value(raw_value: str, column_plan: ColumnPlan) -> Any:
    value = raw_value.strip()
    if value == "":
        return None
    if column_plan.value_kind == "date":
        return date.fromisoformat(value)
    if column_plan.value_kind == "datetime":
        return parse_datetime_value(value)
    return value


def iter_file_rows(table_plan: TablePlan, max_rows_per_file: int | None = None):
    with table_plan.source_file.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter=table_plan.delimiter)
        next(reader, None)

        for row_number, row in enumerate(reader, start=1):
            if len(row) != len(table_plan.columns):
                raise ValueError(
                    f"{table_plan.source_file.name} row {row_number} has {len(row)} columns, "
                    f"expected {len(table_plan.columns)}."
                )

            yield tuple(
                coerce_value(raw_value, column_plan)
                for raw_value, column_plan in zip(row, table_plan.columns, strict=True)
            )

            if max_rows_per_file is not None and row_number >= max_rows_per_file:
                break


def insert_rows_for_table(
    cursor: Any,
    raw_connection: Any,
    schema: str,
    table_plan: TablePlan,
    batch_size: int,
    max_rows_per_file: int | None,
) -> int:
    column_list = ", ".join(quote_identifier(column.name) for column in table_plan.columns)
    placeholders = ", ".join("?" for _ in table_plan.columns)
    insert_sql = (
        f"INSERT INTO {quote_identifier(schema)}.{quote_identifier(table_plan.table_name)} "
        f"({column_list}) VALUES ({placeholders})"
    )

    rows_loaded = 0
    batch: list[tuple[Any, ...]] = []
    for row in iter_file_rows(table_plan, max_rows_per_file=max_rows_per_file):
        batch.append(row)
        if len(batch) < batch_size:
            continue

        cursor.executemany(insert_sql, batch)
        raw_connection.commit()
        rows_loaded += len(batch)
        batch.clear()

    if batch:
        cursor.executemany(insert_sql, batch)
        raw_connection.commit()
        rows_loaded += len(batch)

    return rows_loaded


def load_synthea_data(state: SyntheaLoaderState) -> SyntheaLoaderState:
    engine = state.get("engine") or get_engine()
    target_database = resolve_target_database(engine, state.get("target_database", DEFAULT_DATABASE))
    schema = state.get("schema", DEFAULT_SCHEMA)
    batch_size = int(state.get("batch_size", DEFAULT_BATCH_SIZE))
    max_rows_per_file = state.get("max_rows_per_file")
    execution_results: list[LoadResult] = []

    raw_connection = engine.raw_connection()
    cursor = raw_connection.cursor()
    cursor.fast_executemany = True

    try:
        if target_database:
            cursor.execute(build_use_database_statement(target_database))
        for table_plan in state["table_plans"]:
            rows_loaded = insert_rows_for_table(
                cursor=cursor,
                raw_connection=raw_connection,
                schema=schema,
                table_plan=table_plan,
                batch_size=batch_size,
                max_rows_per_file=max_rows_per_file,
            )
            execution_results.append(
                LoadResult(
                    source_file=table_plan.source_file.name,
                    table_name=table_plan.table_name,
                    status="loaded",
                    rows_loaded=rows_loaded,
                    reason="loaded from Synthea CSV",
                )
            )
    except Exception:
        raw_connection.rollback()
        raise
    finally:
        cursor.close()
        raw_connection.close()

    return {"execution_results": execution_results}


def summarize_run(state: SyntheaLoaderState) -> SyntheaLoaderState:
    execution_results = state.get("execution_results", [])
    rows_by_table = {result.table_name: result.rows_loaded for result in execution_results}

    return {
        "summary": {
            "target_database": resolve_target_database(
                state.get("engine"),
                state.get("target_database", DEFAULT_DATABASE),
            ),
            "schema": state.get("schema", DEFAULT_SCHEMA),
            "table_prefix": state.get("table_prefix", ""),
            "planned_files": len(state.get("table_plans", [])),
            "planned_tables": [table_plan.table_name for table_plan in state.get("table_plans", [])],
            "loaded_files": len(execution_results),
            "rows_loaded": sum(result.rows_loaded for result in execution_results),
            "rows_by_table": rows_by_table,
        }
    }


def build_synthea_loader_graph():
    workflow = StateGraph(SyntheaLoaderState)
    workflow.add_node("discover", discover_synthea_files)
    workflow.add_node("infer", infer_table_plans)
    workflow.add_node("create", create_target_tables)
    workflow.add_node("load", load_synthea_data)
    workflow.add_node("summarize", summarize_run)

    workflow.set_entry_point("discover")
    workflow.add_edge("discover", "infer")
    workflow.add_conditional_edges(
        "infer",
        should_execute_plan,
        {"create": "create", "summarize": "summarize"},
    )
    workflow.add_edge("create", "load")
    workflow.add_edge("load", "summarize")
    workflow.add_edge("summarize", END)
    return workflow.compile()


synthea_loader_agent = build_synthea_loader_graph()


def run_synthea_loader_agent(
    synthea_directory: Path = DEFAULT_SYNTHEA_DIR,
    target_database: str | None = DEFAULT_DATABASE,
    schema: str = DEFAULT_SCHEMA,
    table_prefix: str = "",
    batch_size: int = DEFAULT_BATCH_SIZE,
    sample_rows: int = DEFAULT_SAMPLE_ROWS,
    execute: bool = True,
    max_rows_per_file: int | None = None,
    engine: Any | None = None,
) -> SyntheaLoaderState:
    initial_state: SyntheaLoaderState = {
        "synthea_directory": Path(synthea_directory),
        "target_database": target_database,
        "schema": schema,
        "table_prefix": table_prefix,
        "batch_size": batch_size,
        "sample_rows": sample_rows,
        "execute": execute,
        "max_rows_per_file": max_rows_per_file,
    }
    if engine is not None:
        initial_state["engine"] = engine
    return synthea_loader_agent.invoke(initial_state)


def render_summary(state: SyntheaLoaderState) -> str:
    summary = state.get("summary", {})
    planned_tables = summary.get("planned_tables", [])
    rows_by_table = summary.get("rows_by_table", {})
    target_database = summary.get("target_database") or "connected database"
    return "\n".join(
        [
            f"Database: {target_database}",
            f"Schema: {summary.get('schema', DEFAULT_SCHEMA)}",
            f"Table prefix: {summary.get('table_prefix', '') or '(none)'}",
            f"Planned files: {summary.get('planned_files', 0)}",
            (
                "Tables: " + ", ".join(planned_tables)
                if planned_tables
                else "Tables: none"
            ),
            f"Loaded files: {summary.get('loaded_files', 0)}",
            f"Rows loaded: {summary.get('rows_loaded', 0)}",
            (
                "Rows by table: "
                + ", ".join(f"{table}={count}" for table, count in rows_by_table.items())
                if rows_by_table
                else "Rows by table: none"
            ),
        ]
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create SQL Server tables from the CSVs in data/synthea_tables under the "
            "requested schema, optionally prefix the table names, and load them into the "
            "target database."
        )
    )
    parser.add_argument(
        "--synthea-directory",
        type=Path,
        default=DEFAULT_SYNTHEA_DIR,
        help="Directory containing the Synthea CSV files.",
    )
    parser.add_argument(
        "--database",
        default=DEFAULT_DATABASE,
        help="Target SQL Server database. Defaults to the current connection database.",
    )
    parser.add_argument(
        "--schema",
        default=DEFAULT_SCHEMA,
        help="Target SQL Server schema. Defaults to synthea.",
    )
    parser.add_argument(
        "--table-prefix",
        default="",
        help="Optional prefix to prepend to each created table name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of rows to insert per batch.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=DEFAULT_SAMPLE_ROWS,
        help="Number of data rows to sample when inferring SQL column types.",
    )
    parser.add_argument(
        "--max-rows-per-file",
        type=int,
        default=None,
        help="Optional cap for sampled runs or tests.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the table plan without creating tables or loading data.",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    final_state = run_synthea_loader_agent(
        synthea_directory=args.synthea_directory,
        target_database=args.database,
        schema=args.schema,
        table_prefix=args.table_prefix,
        batch_size=args.batch_size,
        sample_rows=args.sample_rows,
        execute=not args.dry_run,
        max_rows_per_file=args.max_rows_per_file,
    )
    print(render_summary(final_state))


if __name__ == "__main__":
    main()
