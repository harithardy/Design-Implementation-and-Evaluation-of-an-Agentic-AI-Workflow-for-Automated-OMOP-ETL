from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph
from sqlalchemy import text

try:
    from src.load_tables import (
        DEFAULT_DATA_DIR,
        DEFAULT_DATABASE,
        DEFAULT_SCHEMA,
        PlannedStatement,
        build_planned_statement,
        normalize_statement,
        object_exists,
        quote_identifier,
        split_sql_statements,
    )
except ModuleNotFoundError:
    from load_tables import (
        DEFAULT_DATA_DIR,
        DEFAULT_DATABASE,
        DEFAULT_SCHEMA,
        PlannedStatement,
        build_planned_statement,
        normalize_statement,
        object_exists,
        quote_identifier,
        split_sql_statements,
    )


@dataclass(frozen=True)
class TruncationResult:
    table_name: str
    status: Literal["emptied", "skipped"]
    rows_before: int
    rows_after: int
    reason: str


class TruncatingState(TypedDict, total=False):
    ddl_directory: Path
    target_database: str
    schema: str
    execute: bool
    engine: Any
    ddl_files: list[Path]
    planned_tables: list[PlannedStatement]
    execution_results: list[TruncationResult]
    summary: dict[str, Any]


def get_engine() -> Any:
    try:
        from src.connection import engine
    except ModuleNotFoundError:
        from connection import engine

    return engine


def discover_ddl_files(state: TruncatingState) -> TruncatingState:
    ddl_directory = Path(state.get("ddl_directory", DEFAULT_DATA_DIR))
    ddl_files = sorted(ddl_directory.glob("*_ddl.sql"), key=lambda path: path.name.lower())
    if not ddl_files:
        raise FileNotFoundError(f"No OMOP DDL SQL files were found in {ddl_directory}.")
    return {"ddl_files": ddl_files}


def plan_omop_tables(state: TruncatingState) -> TruncatingState:
    schema = state.get("schema", DEFAULT_SCHEMA)
    planned_tables: list[PlannedStatement] = []
    seen_tables: set[str] = set()

    for ddl_file in state["ddl_files"]:
        sql_text = ddl_file.read_text(encoding="utf-8")
        for raw_statement in split_sql_statements(sql_text):
            planned_statement = build_planned_statement(
                statement=normalize_statement(raw_statement, schema),
                phase="ddl",
                source_file=ddl_file.name,
                schema=schema,
            )
            if planned_statement.object_type != "table":
                continue
            if planned_statement.table_name in seen_tables:
                continue
            seen_tables.add(planned_statement.table_name)
            planned_tables.append(planned_statement)

    if not planned_tables:
        raise ValueError("No OMOP tables were found in the supplied DDL files.")

    return {"planned_tables": planned_tables}


def should_execute_plan(state: TruncatingState) -> str:
    return "execute" if state.get("execute", True) else "summarize"


def count_rows(connection: Any, table_name: str) -> int:
    result = connection.execute(text(f"SELECT COUNT_BIG(1) AS row_count FROM {table_name}"))
    value = result.scalar()
    return int(value or 0)


def empty_omop_tables(state: TruncatingState) -> TruncatingState:
    engine = state.get("engine") or get_engine()
    target_database = state.get("target_database", DEFAULT_DATABASE)
    execution_results: list[TruncationResult] = []
    existing_tables: list[tuple[PlannedStatement, int]] = []

    with engine.connect() as connection:
        connection.execute(text(f"USE {quote_identifier(target_database)}"))
        for planned_table in state["planned_tables"]:
            if not object_exists(connection, planned_table):
                execution_results.append(
                    TruncationResult(
                        table_name=planned_table.table_name,
                        status="skipped",
                        rows_before=0,
                        rows_after=0,
                        reason="table not found",
                    )
                )
                continue
            existing_tables.append((planned_table, count_rows(connection, planned_table.table_name)))

        disabled_tables: list[PlannedStatement] = []
        try:
            for planned_table, _ in existing_tables:
                connection.execute(text(f"ALTER TABLE {planned_table.table_name} NOCHECK CONSTRAINT ALL"))
                disabled_tables.append(planned_table)

            for planned_table, _ in existing_tables:
                connection.execute(text(f"DELETE FROM {planned_table.table_name}"))
        finally:
            for planned_table in reversed(disabled_tables):
                connection.execute(
                    text(f"ALTER TABLE {planned_table.table_name} WITH CHECK CHECK CONSTRAINT ALL")
                )

        for planned_table, rows_before in existing_tables:
            rows_after = count_rows(connection, planned_table.table_name)
            if rows_after != 0:
                raise RuntimeError(
                    f"{planned_table.table_name} still contains {rows_after} rows after truncation."
                )
            execution_results.append(
                TruncationResult(
                    table_name=planned_table.table_name,
                    status="emptied",
                    rows_before=rows_before,
                    rows_after=rows_after,
                    reason="table cleared",
                )
            )

    return {"execution_results": execution_results}


def summarize_run(state: TruncatingState) -> TruncatingState:
    planned_tables = state.get("planned_tables", [])
    execution_results = state.get("execution_results", [])
    return {
        "summary": {
            "target_database": state.get("target_database", DEFAULT_DATABASE),
            "schema": state.get("schema", DEFAULT_SCHEMA),
            "tables_planned": len(planned_tables),
            "planned_table_names": [planned_table.table_name for planned_table in planned_tables],
            "tables_emptied": sum(1 for result in execution_results if result.status == "emptied"),
            "tables_skipped": sum(1 for result in execution_results if result.status == "skipped"),
            "rows_removed": sum(result.rows_before - result.rows_after for result in execution_results),
        }
    }


def build_truncating_graph():
    workflow = StateGraph(TruncatingState)
    workflow.add_node("discover", discover_ddl_files)
    workflow.add_node("plan", plan_omop_tables)
    workflow.add_node("execute", empty_omop_tables)
    workflow.add_node("summarize", summarize_run)

    workflow.set_entry_point("discover")
    workflow.add_edge("discover", "plan")
    workflow.add_conditional_edges(
        "plan",
        should_execute_plan,
        {"execute": "execute", "summarize": "summarize"},
    )
    workflow.add_edge("execute", "summarize")
    workflow.add_edge("summarize", END)
    return workflow.compile()


truncating_agent = build_truncating_graph()


def run_truncating_agent(
    ddl_directory: Path = DEFAULT_DATA_DIR,
    target_database: str = DEFAULT_DATABASE,
    schema: str = DEFAULT_SCHEMA,
    execute: bool = True,
    engine: Any | None = None,
) -> TruncatingState:
    initial_state: TruncatingState = {
        "ddl_directory": Path(ddl_directory),
        "target_database": target_database,
        "schema": schema,
        "execute": execute,
    }
    if engine is not None:
        initial_state["engine"] = engine
    return truncating_agent.invoke(initial_state)


def render_summary(state: TruncatingState) -> str:
    summary = state.get("summary", {})
    planned_table_names = summary.get("planned_table_names", [])
    return "\n".join(
        [
            f"Database: {summary.get('target_database', DEFAULT_DATABASE)}",
            f"Schema: {summary.get('schema', DEFAULT_SCHEMA)}",
            f"Tables planned: {summary.get('tables_planned', 0)}",
            (
                "Tables: " + ", ".join(planned_table_names)
                if planned_table_names
                else "Tables: none"
            ),
            f"Tables emptied: {summary.get('tables_emptied', 0)}",
            f"Tables skipped: {summary.get('tables_skipped', 0)}",
            f"Rows removed: {summary.get('rows_removed', 0)}",
        ]
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Empty the OMOP tables defined by the OMOP *_ddl.sql file in AI_OMOP while "
            "leaving non-OMOP tables such as synthea tables untouched."
        )
    )
    parser.add_argument(
        "--ddl-directory",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing the OMOP SQL files.",
    )
    parser.add_argument(
        "--database",
        default=DEFAULT_DATABASE,
        help="Target SQL Server database. Defaults to AI_OMOP.",
    )
    parser.add_argument(
        "--schema",
        default=DEFAULT_SCHEMA,
        help="Target SQL Server schema. Defaults to dbo.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan the affected OMOP tables without deleting any rows.",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    final_state = run_truncating_agent(
        ddl_directory=args.ddl_directory,
        target_database=args.database,
        schema=args.schema,
        execute=not args.dry_run,
    )
    print(render_summary(final_state))


if __name__ == "__main__":
    main()
