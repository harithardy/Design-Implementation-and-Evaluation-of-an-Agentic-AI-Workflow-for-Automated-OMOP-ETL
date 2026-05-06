from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph
from sqlalchemy import text

try:
    from src.load_tables import DEFAULT_DATA_DIR, DEFAULT_DATABASE, DEFAULT_SCHEMA, quote_identifier
except ModuleNotFoundError:
    from load_tables import DEFAULT_DATA_DIR, DEFAULT_DATABASE, DEFAULT_SCHEMA, quote_identifier

DEFAULT_VOCABULARY_DIR = DEFAULT_DATA_DIR / "vocabulary_athena"
DEFAULT_BULK_ROW_TERMINATOR = "0x0A"
DEFAULT_BULK_FIELD_TERMINATOR = "\\t"

TABLE_LOAD_ORDER = (
    "domain",
    "vocabulary",
    "concept_class",
    "relationship",
    "concept",
    "concept_synonym",
    "concept_relationship",
    "drug_strength",
    "concept_ancestor",
)

TABLE_DELETE_ORDER = tuple(reversed(TABLE_LOAD_ORDER))

EXACT_FILE_MAP = {
    "domain": ("DOMAIN.csv",),
    "vocabulary": ("VOCABULARY.csv",),
    "concept_class": ("CONCEPT_CLASS.csv",),
    "relationship": ("RELATIONSHIP.csv",),
    "concept_synonym": ("CONCEPT_SYNONYM.csv",),
    "concept_relationship": ("CONCEPT_RELATIONSHIP.csv",),
    "drug_strength": ("DRUG_STRENGTH.csv",),
    "concept_ancestor": ("CONCEPT_ANCESTOR.csv",),
}

CONCEPT_EXCLUDED_FILES = {
    "CONCEPT_CLASS.CSV",
    "CONCEPT_SYNONYM.CSV",
    "CONCEPT_RELATIONSHIP.CSV",
    "CONCEPT_ANCESTOR.CSV",
}


@dataclass(frozen=True)
class VocabularyBulkLoadPlan:
    table_name: str
    source_files: tuple[Path, ...]


@dataclass(frozen=True)
class VocabularyBulkLoadResult:
    table_name: str
    status: Literal["loaded", "skipped"]
    source_files: tuple[str, ...]
    rows_loaded: int
    reason: str


class VocabularyLoadingState(TypedDict, total=False):
    vocabulary_directory: Path
    target_database: str
    schema: str
    execute: bool
    engine: Any
    vocabulary_files: list[Path]
    ignored_files: list[str]
    load_plans: list[VocabularyBulkLoadPlan]
    execution_results: list[VocabularyBulkLoadResult]
    summary: dict[str, Any]


def get_engine() -> Any:
    try:
        from src.connection import engine
    except ModuleNotFoundError:
        from connection import engine

    return engine


def discover_vocabulary_files(state: VocabularyLoadingState) -> VocabularyLoadingState:
    vocabulary_directory = Path(state.get("vocabulary_directory", DEFAULT_VOCABULARY_DIR))
    vocabulary_files = sorted(vocabulary_directory.glob("*.csv"), key=lambda path: path.name.lower())
    if not vocabulary_files:
        raise FileNotFoundError(f"No vocabulary CSV files were found in {vocabulary_directory}.")
    return {"vocabulary_files": vocabulary_files}


def is_concept_source_file(file_path: Path) -> bool:
    upper_name = file_path.name.upper()
    return upper_name == "CONCEPT.CSV" or (
        upper_name.startswith("CONCEPT_") and upper_name not in CONCEPT_EXCLUDED_FILES
    )


def find_source_files_for_table(table_name: str, vocabulary_files: list[Path]) -> tuple[Path, ...]:
    if table_name == "concept":
        return tuple(
            sorted(
                [path for path in vocabulary_files if is_concept_source_file(path)],
                key=lambda path: path.name.lower(),
            )
        )

    expected_names = {name.upper() for name in EXACT_FILE_MAP[table_name]}
    return tuple(
        path
        for path in sorted(vocabulary_files, key=lambda item: item.name.lower())
        if path.name.upper() in expected_names
    )


def plan_vocabulary_loads(state: VocabularyLoadingState) -> VocabularyLoadingState:
    vocabulary_files = state["vocabulary_files"]
    load_plans = [
        VocabularyBulkLoadPlan(
            table_name=table_name,
            source_files=find_source_files_for_table(table_name, vocabulary_files),
        )
        for table_name in TABLE_LOAD_ORDER
    ]

    used_files = {
        source_file.resolve()
        for load_plan in load_plans
        for source_file in load_plan.source_files
    }
    ignored_files = [
        file_path.name
        for file_path in vocabulary_files
        if file_path.resolve() not in used_files
    ]
    if ignored_files:
        ignored_list = ", ".join(sorted(ignored_files))
        raise ValueError(f"Unmapped vocabulary CSV files were found: {ignored_list}")

    return {"load_plans": load_plans, "ignored_files": ignored_files}


def should_execute_plan(state: VocabularyLoadingState) -> str:
    return "execute" if state.get("execute", True) else "summarize"


def build_table_name(schema: str, table_name: str) -> str:
    return f"{quote_identifier(schema)}.{quote_identifier(table_name)}"


def sql_string_literal(value: str) -> str:
    return value.replace("'", "''")


def build_bulk_insert_statement(schema: str, table_name: str, file_path: Path) -> str:
    resolved_path = sql_string_literal(str(file_path.resolve()))
    return "\n".join(
        [
            f"BULK INSERT {build_table_name(schema, table_name)}",
            f"FROM N'{resolved_path}'",
            "WITH (",
            f"    FIELDTERMINATOR = '{DEFAULT_BULK_FIELD_TERMINATOR}',",
            f"    ROWTERMINATOR = '{DEFAULT_BULK_ROW_TERMINATOR}',",
            "    FIRSTROW = 2,",
            "    CODEPAGE = '65001',",
            "    TABLOCK",
            ")",
        ]
    )


def build_delete_statement(schema: str, table_name: str) -> str:
    return f"DELETE FROM {build_table_name(schema, table_name)}"


def build_count_statement(schema: str, table_name: str) -> str:
    return f"SELECT COUNT(*) FROM {build_table_name(schema, table_name)}"


def build_enable_constraints_statement(schema: str, table_name: str) -> str:
    return f"ALTER TABLE {build_table_name(schema, table_name)} CHECK CONSTRAINT ALL"


def count_rows(connection: Any, schema: str, table_name: str) -> int:
    return int(connection.execute(text(build_count_statement(schema, table_name))).scalar() or 0)


def execute_vocabulary_loads(state: VocabularyLoadingState) -> VocabularyLoadingState:
    engine = state.get("engine") or get_engine()
    target_database = state.get("target_database", DEFAULT_DATABASE)
    schema = state.get("schema", DEFAULT_SCHEMA)
    execution_results: list[VocabularyBulkLoadResult] = []

    with engine.connect() as connection:
        connection.execute(text(f"USE {quote_identifier(target_database)}"))

        for load_plan in state["load_plans"]:
            connection.execute(text(build_enable_constraints_statement(schema, load_plan.table_name)))

        for table_name in TABLE_DELETE_ORDER:
            connection.execute(text(build_delete_statement(schema, table_name)))

        for load_plan in state["load_plans"]:
            if not load_plan.source_files:
                rows_loaded = count_rows(connection, schema, load_plan.table_name)
                execution_results.append(
                    VocabularyBulkLoadResult(
                        table_name=load_plan.table_name,
                        status="skipped",
                        source_files=(),
                        rows_loaded=rows_loaded,
                        reason="no source file found",
                    )
                )
                continue

            for source_file in load_plan.source_files:
                connection.execute(
                    text(build_bulk_insert_statement(schema, load_plan.table_name, source_file))
                )

            rows_loaded = count_rows(connection, schema, load_plan.table_name)
            execution_results.append(
                VocabularyBulkLoadResult(
                    table_name=load_plan.table_name,
                    status="loaded",
                    source_files=tuple(source_file.name for source_file in load_plan.source_files),
                    rows_loaded=rows_loaded,
                    reason="bulk insert completed",
                )
            )

    return {"execution_results": execution_results}


def summarize_run(state: VocabularyLoadingState) -> VocabularyLoadingState:
    execution_results = state.get("execution_results", [])
    return {
        "summary": {
            "target_database": state.get("target_database", DEFAULT_DATABASE),
            "schema": state.get("schema", DEFAULT_SCHEMA),
            "tables_planned": [load_plan.table_name for load_plan in state.get("load_plans", [])],
            "ignored_files": state.get("ignored_files", []),
            "tables_loaded": sum(1 for result in execution_results if result.status == "loaded"),
            "tables_skipped": sum(1 for result in execution_results if result.status == "skipped"),
            "rows_by_table": {
                result.table_name: result.rows_loaded for result in execution_results
            },
        }
    }


def build_vocabulary_loading_graph():
    workflow = StateGraph(VocabularyLoadingState)
    workflow.add_node("discover", discover_vocabulary_files)
    workflow.add_node("plan", plan_vocabulary_loads)
    workflow.add_node("execute", execute_vocabulary_loads)
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


vocabulary_loading_agent = build_vocabulary_loading_graph()


def run_vocabulary_loading_agent(
    vocabulary_directory: Path = DEFAULT_VOCABULARY_DIR,
    target_database: str = DEFAULT_DATABASE,
    schema: str = DEFAULT_SCHEMA,
    execute: bool = True,
    engine: Any | None = None,
) -> VocabularyLoadingState:
    initial_state: VocabularyLoadingState = {
        "vocabulary_directory": Path(vocabulary_directory),
        "target_database": target_database,
        "schema": schema,
        "execute": execute,
    }
    if engine is not None:
        initial_state["engine"] = engine
    return vocabulary_loading_agent.invoke(initial_state)


def render_summary(state: VocabularyLoadingState) -> str:
    summary = state.get("summary", {})
    rows_by_table = summary.get("rows_by_table", {})
    return "\n".join(
        [
            f"Database: {summary.get('target_database', DEFAULT_DATABASE)}",
            f"Schema: {summary.get('schema', DEFAULT_SCHEMA)}",
            (
                "Planned tables: " + ", ".join(summary.get("tables_planned", []))
                if summary.get("tables_planned")
                else "Planned tables: none"
            ),
            (
                "Ignored files: " + ", ".join(summary.get("ignored_files", []))
                if summary.get("ignored_files")
                else "Ignored files: none"
            ),
            f"Tables loaded: {summary.get('tables_loaded', 0)}",
            f"Tables skipped: {summary.get('tables_skipped', 0)}",
            (
                "Rows by table: " + ", ".join(
                    f"{table_name}={count}" for table_name, count in rows_by_table.items()
                )
                if rows_by_table
                else "Rows by table: none"
            ),
        ]
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load the Athena vocabulary CSV files from data/vocabulary_athena into the "
            "requested OMOP vocabulary tables using SQL Server BULK INSERT."
        )
    )
    parser.add_argument(
        "--vocabulary-directory",
        type=Path,
        default=DEFAULT_VOCABULARY_DIR,
        help="Directory containing the Athena vocabulary CSV files.",
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
        help="Plan the BULK INSERT operations without modifying the database.",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    final_state = run_vocabulary_loading_agent(
        vocabulary_directory=args.vocabulary_directory,
        target_database=args.database,
        schema=args.schema,
        execute=not args.dry_run,
    )
    print(render_summary(final_state))


if __name__ == "__main__":
    main()
