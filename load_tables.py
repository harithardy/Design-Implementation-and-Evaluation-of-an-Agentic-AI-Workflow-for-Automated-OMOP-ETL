from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DATABASE = "AI_OMOP"
DEFAULT_SCHEMA = "dbo"

COMMENT_BLOCK_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
COMMENT_LINE_RE = re.compile(r"--.*?$", re.MULTILINE)
SCHEMA_TOKEN_RE = re.compile(r"@cdmDatabaseSchema\.([A-Za-z0-9_]+)", re.IGNORECASE)
CREATE_TABLE_RE = re.compile(r"CREATE\s+TABLE\s+([^\s(]+)", re.IGNORECASE)
ALTER_CONSTRAINT_RE = re.compile(
    r"ALTER\s+TABLE\s+([^\s]+)\s+ADD\s+CONSTRAINT\s+([^\s]+)",
    re.IGNORECASE,
)
CREATE_INDEX_RE = re.compile(
    r"CREATE(?:\s+CLUSTERED|\s+NONCLUSTERED)?\s+INDEX\s+([^\s]+)\s+ON\s+([^\s(]+)",
    re.IGNORECASE,
)

FILE_PHASE_ORDER = {
    "_ddl.sql": ("ddl", 0),
    "_primary_keys.sql": ("primary_keys", 1),
    "_constraints.sql": ("constraints", 2),
    "_indices.sql": ("indices", 3),
}


@dataclass(frozen=True)
class PlannedStatement:
    phase: Literal["ddl", "primary_keys", "constraints", "indices"]
    statement: str
    object_type: Literal["table", "constraint", "index"]
    object_name: str
    table_name: str
    source_file: str


@dataclass(frozen=True)
class ExecutionResult:
    phase: Literal["ddl", "primary_keys", "constraints", "indices"]
    object_type: Literal["table", "constraint", "index"]
    object_name: str
    source_file: str
    status: Literal["executed", "skipped"]
    reason: str


class LoadTablesState(TypedDict, total=False):
    ddl_directory: Path
    target_database: str
    schema: str
    include_indices: bool
    execute: bool
    engine: Any
    sql_files: list[Path]
    planned_statements: list[PlannedStatement]
    execution_results: list[ExecutionResult]
    summary: dict[str, Any]


def quote_identifier(identifier: str) -> str:
    return f"[{identifier.replace(']', ']]')}]"


def split_qualified_name(name: str, default_schema: str) -> tuple[str, str]:
    parts = [part.strip().strip("[]") for part in name.split(".") if part.strip()]
    if not parts:
        raise ValueError("SQL object name cannot be empty.")
    if len(parts) == 1:
        return default_schema, parts[0]
    return parts[-2], parts[-1]


def canonical_table_name(name: str, default_schema: str) -> str:
    schema_name, table_name = split_qualified_name(name, default_schema)
    return f"{quote_identifier(schema_name)}.{quote_identifier(table_name)}"


def strip_sql_comments(sql_text: str) -> str:
    without_blocks = COMMENT_BLOCK_RE.sub("", sql_text)
    return COMMENT_LINE_RE.sub("", without_blocks)


def split_sql_statements(sql_text: str) -> list[str]:
    stripped_text = strip_sql_comments(sql_text)
    return [statement.strip() for statement in stripped_text.split(";") if statement.strip()]


def normalize_statement(statement: str, schema: str) -> str:
    normalized = SCHEMA_TOKEN_RE.sub(
        lambda match: f"{quote_identifier(schema)}.{quote_identifier(match.group(1))}",
        statement,
    )
    return normalized.strip()


def classify_sql_file(sql_file: Path) -> tuple[str, int]:
    lower_name = sql_file.name.lower()
    for suffix, (phase, order) in FILE_PHASE_ORDER.items():
        if lower_name.endswith(suffix):
            return phase, order
    return "ddl", len(FILE_PHASE_ORDER)


def build_planned_statement(
    statement: str,
    phase: str,
    source_file: str,
    schema: str,
) -> PlannedStatement:
    create_table_match = CREATE_TABLE_RE.search(statement)
    if create_table_match:
        table_name = canonical_table_name(create_table_match.group(1), schema)
        return PlannedStatement(
            phase=phase,
            statement=statement,
            object_type="table",
            object_name=table_name,
            table_name=table_name,
            source_file=source_file,
        )

    constraint_match = ALTER_CONSTRAINT_RE.search(statement)
    if constraint_match:
        table_name = canonical_table_name(constraint_match.group(1), schema)
        return PlannedStatement(
            phase=phase,
            statement=statement,
            object_type="constraint",
            object_name=constraint_match.group(2).strip("[]"),
            table_name=table_name,
            source_file=source_file,
        )

    index_match = CREATE_INDEX_RE.search(statement)
    if index_match:
        table_name = canonical_table_name(index_match.group(2), schema)
        return PlannedStatement(
            phase=phase,
            statement=statement,
            object_type="index",
            object_name=index_match.group(1).strip("[]"),
            table_name=table_name,
            source_file=source_file,
        )

    preview = statement.replace("\n", " ")[:120]
    raise ValueError(f"Unsupported SQL statement in {source_file}: {preview}")


def get_engine() -> Any:
    try:
        from src.connection import engine
    except ModuleNotFoundError:
        from connection import engine

    return engine


def discover_sql_files(state: LoadTablesState) -> LoadTablesState:
    ddl_directory = Path(state.get("ddl_directory", DEFAULT_DATA_DIR))
    sql_files = sorted(
        ddl_directory.glob("*.sql"),
        key=lambda path: (classify_sql_file(path)[1], path.name.lower()),
    )
    if not sql_files:
        raise FileNotFoundError(f"No SQL files were found in {ddl_directory}.")
    return {"sql_files": sql_files}


def plan_sql_statements(state: LoadTablesState) -> LoadTablesState:
    schema = state.get("schema", DEFAULT_SCHEMA)
    include_indices = state.get("include_indices", True)
    planned_statements: list[PlannedStatement] = []

    for sql_file in state["sql_files"]:
        phase, _ = classify_sql_file(sql_file)
        if phase == "indices" and not include_indices:
            continue

        sql_text = sql_file.read_text(encoding="utf-8")
        for raw_statement in split_sql_statements(sql_text):
            normalized_statement = normalize_statement(raw_statement, schema)
            planned_statements.append(
                build_planned_statement(
                    statement=normalized_statement,
                    phase=phase,
                    source_file=sql_file.name,
                    schema=schema,
                )
            )

    return {"planned_statements": planned_statements}


def should_execute_plan(state: LoadTablesState) -> str:
    return "execute" if state.get("execute", True) else "summarize"


def object_exists(connection: Any, statement: PlannedStatement) -> bool:
    if statement.object_type == "table":
        schema_name, table_name = split_qualified_name(statement.table_name, DEFAULT_SCHEMA)
        result = connection.execute(
            text(
                """
                SELECT 1
                FROM sys.tables AS tables
                INNER JOIN sys.schemas AS schemas
                    ON schemas.schema_id = tables.schema_id
                WHERE schemas.name = :schema_name
                  AND tables.name = :table_name
                """
            ),
            {"schema_name": schema_name, "table_name": table_name},
        )
        return result.scalar() is not None

    if statement.object_type == "constraint":
        result = connection.execute(
            text(
                """
                SELECT 1
                FROM sys.objects
                WHERE name = :object_name
                  AND parent_object_id = OBJECT_ID(:table_name)
                """
            ),
            {"object_name": statement.object_name, "table_name": statement.table_name},
        )
        return result.scalar() is not None

    result = connection.execute(
        text(
            """
            SELECT 1
            FROM sys.indexes
            WHERE name = :object_name
              AND object_id = OBJECT_ID(:table_name)
            """
        ),
        {"object_name": statement.object_name, "table_name": statement.table_name},
    )
    return result.scalar() is not None


def execute_sql_plan(state: LoadTablesState) -> LoadTablesState:
    target_database = state.get("target_database", DEFAULT_DATABASE)
    execution_results: list[ExecutionResult] = []
    engine = state.get("engine") or get_engine()

    with engine.connect() as connection:
        connection.execute(text(f"USE {quote_identifier(target_database)}"))
        for planned_statement in state["planned_statements"]:
            if object_exists(connection, planned_statement):
                execution_results.append(
                    ExecutionResult(
                        phase=planned_statement.phase,
                        object_type=planned_statement.object_type,
                        object_name=planned_statement.object_name,
                        source_file=planned_statement.source_file,
                        status="skipped",
                        reason="already exists",
                    )
                )
                continue

            connection.execute(text(planned_statement.statement))
            execution_results.append(
                ExecutionResult(
                    phase=planned_statement.phase,
                    object_type=planned_statement.object_type,
                    object_name=planned_statement.object_name,
                    source_file=planned_statement.source_file,
                    status="executed",
                    reason="created",
                )
            )

    return {"execution_results": execution_results}


def summarize_run(state: LoadTablesState) -> LoadTablesState:
    planned_statements = state.get("planned_statements", [])
    execution_results = state.get("execution_results", [])
    planned_counts = Counter(statement.phase for statement in planned_statements)
    status_counts = Counter(result.status for result in execution_results)

    return {
        "summary": {
            "target_database": state.get("target_database", DEFAULT_DATABASE),
            "schema": state.get("schema", DEFAULT_SCHEMA),
            "planned_statements": len(planned_statements),
            "planned_by_phase": dict(planned_counts),
            "executed": status_counts.get("executed", 0),
            "skipped": status_counts.get("skipped", 0),
        }
    }


def build_load_tables_graph():
    workflow = StateGraph(LoadTablesState)
    workflow.add_node("discover", discover_sql_files)
    workflow.add_node("plan", plan_sql_statements)
    workflow.add_node("execute", execute_sql_plan)
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


load_tables = build_load_tables_graph()


def run_load_tables(
    ddl_directory: Path = DEFAULT_DATA_DIR,
    target_database: str = DEFAULT_DATABASE,
    schema: str = DEFAULT_SCHEMA,
    include_indices: bool = True,
    execute: bool = True,
    engine: Any | None = None,
) -> LoadTablesState:
    initial_state: LoadTablesState = {
        "ddl_directory": Path(ddl_directory),
        "target_database": target_database,
        "schema": schema,
        "include_indices": include_indices,
        "execute": execute,
    }
    if engine is not None:
        initial_state["engine"] = engine
    return load_tables.invoke(initial_state)


def render_summary(state: LoadTablesState) -> str:
    summary = state.get("summary", {})
    planned_by_phase = summary.get("planned_by_phase", {})
    lines = [
        f"Database: {summary.get('target_database', DEFAULT_DATABASE)}",
        f"Schema: {summary.get('schema', DEFAULT_SCHEMA)}",
        f"Planned statements: {summary.get('planned_statements', 0)}",
        (
            "Phase counts: "
            + ", ".join(f"{phase}={count}" for phase, count in planned_by_phase.items())
            if planned_by_phase
            else "Phase counts: none"
        ),
        f"Executed: {summary.get('executed', 0)}",
        f"Skipped: {summary.get('skipped', 0)}",
    ]
    return "\n".join(lines)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read OMOP SQL DDL files and create the standard tables in AI_OMOP."
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
        "--skip-indices",
        action="store_true",
        help="Skip the index creation SQL file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan the SQL execution without creating any tables, constraints, or indexes.",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    final_state = run_load_tables(
        ddl_directory=args.ddl_directory,
        target_database=args.database,
        schema=args.schema,
        include_indices=not args.skip_indices,
        execute=not args.dry_run,
    )
    print(render_summary(final_state))


if __name__ == "__main__":
    main()
