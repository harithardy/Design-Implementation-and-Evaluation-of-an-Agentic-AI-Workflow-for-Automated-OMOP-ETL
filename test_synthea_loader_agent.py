from datetime import date, datetime
from pathlib import Path

from src.synthea_loader_agent import (
    ColumnPlan,
    DEFAULT_SCHEMA,
    TablePlan,
    build_target_table_name,
    build_create_table_statement,
    coerce_value,
    create_target_tables,
    discover_synthea_files,
    infer_table_plans,
    load_synthea_data,
)


def write_csv(path: Path, content: str) -> None:
    path.write_text(content.strip() + "\n", encoding="utf-8")


def test_infer_table_plans_detects_uuid_and_temporal_columns(tmp_path: Path) -> None:
    write_csv(
        tmp_path / "patients.csv",
        """
        Id,BIRTHDATE,LAST_UPDATE,FIRST
        b9c610cd-28a6-4636-ccb6-c7a0d2a4cb85,2019-02-17,2019-02-17T05:07:38Z,Damon
        """,
    )

    discovered_state = discover_synthea_files({"synthea_directory": tmp_path})
    planned_state = infer_table_plans(
        {**discovered_state, "sample_rows": 10}
    )

    table_plan = planned_state["table_plans"][0]
    assert [column.sql_type for column in table_plan.columns] == [
        "UNIQUEIDENTIFIER",
        "DATE",
        "DATETIME2",
        "NVARCHAR(MAX)",
    ]


def test_build_create_table_statement_uses_inferred_schema() -> None:
    table_plan = TablePlan(
        source_file=Path("patients.csv"),
        table_name="patients",
        delimiter=",",
        columns=(
            ColumnPlan("Id", "UNIQUEIDENTIFIER", "uuid"),
            ColumnPlan("BIRTHDATE", "DATE", "date"),
            ColumnPlan("FIRST", "NVARCHAR(MAX)", "text"),
        ),
    )

    statement = build_create_table_statement(table_plan, DEFAULT_SCHEMA)
    assert statement == (
        "CREATE TABLE [synthea].[patients] "
        "([Id] UNIQUEIDENTIFIER NULL, [BIRTHDATE] DATE NULL, [FIRST] NVARCHAR(MAX) NULL)"
    )


def test_build_target_table_name_applies_optional_prefix() -> None:
    assert build_target_table_name(Path("patients.csv")) == "patients"
    assert build_target_table_name(Path("patients.csv"), "synthea_") == "synthea_patients"


class FakeConnection:
    def __init__(self) -> None:
        self.executed: list[tuple[str, object | None]] = []

    def execute(self, statement: object, params: object | None = None):
        self.executed.append((str(statement), params))
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeCreateEngine:
    def __init__(self, database: str | None = None) -> None:
        self.connection = FakeConnection()
        self.url = type("FakeUrl", (), {"database": database})()

    def connect(self):
        return self.connection


def test_create_target_tables_uses_synthea_schema_in_connected_database() -> None:
    fake_engine = FakeCreateEngine(database="ConnectedWarehouse")
    create_target_tables(
        {
            "engine": fake_engine,
            "table_plans": [
                TablePlan(
                    source_file=Path("patients.csv"),
                    table_name="patients",
                    delimiter=",",
                    columns=(
                        ColumnPlan("Id", "UNIQUEIDENTIFIER", "uuid"),
                        ColumnPlan("FIRST", "NVARCHAR(MAX)", "text"),
                    ),
                )
            ],
        }
    )

    executed_sql = [sql for sql, _ in fake_engine.connection.executed]
    executed_params = [params for _, params in fake_engine.connection.executed]

    assert executed_sql == [
        "USE [ConnectedWarehouse]",
        "IF SCHEMA_ID(:schema_name) IS NULL EXEC(N'CREATE SCHEMA [synthea]')",
        "DROP TABLE IF EXISTS [synthea].[patients]",
        "CREATE TABLE [synthea].[patients] ([Id] UNIQUEIDENTIFIER NULL, [FIRST] NVARCHAR(MAX) NULL)",
    ]
    assert executed_params[1] == {"schema_name": "synthea"}


def test_create_target_tables_supports_dbo_prefixed_table_names() -> None:
    fake_engine = FakeCreateEngine(database="AI_OMOP")
    create_target_tables(
        {
            "engine": fake_engine,
            "schema": "dbo",
            "table_plans": [
                TablePlan(
                    source_file=Path("patients.csv"),
                    table_name="synthea_patients",
                    delimiter=",",
                    columns=(
                        ColumnPlan("Id", "UNIQUEIDENTIFIER", "uuid"),
                        ColumnPlan("FIRST", "NVARCHAR(MAX)", "text"),
                    ),
                )
            ],
        }
    )

    executed_sql = [sql for sql, _ in fake_engine.connection.executed]
    assert executed_sql == [
        "USE [AI_OMOP]",
        "IF SCHEMA_ID(:schema_name) IS NULL EXEC(N'CREATE SCHEMA [dbo]')",
        "DROP TABLE IF EXISTS [dbo].[synthea_patients]",
        "CREATE TABLE [dbo].[synthea_patients] ([Id] UNIQUEIDENTIFIER NULL, [FIRST] NVARCHAR(MAX) NULL)",
    ]


class FakeCursor:
    def __init__(self) -> None:
        self.fast_executemany = False
        self.executed_sql: list[str] = []
        self.executemany_calls: list[tuple[str, list[tuple[object, ...]]]] = []

    def execute(self, sql: str):
        self.executed_sql.append(sql)
        return self

    def executemany(self, sql: str, rows: list[tuple[object, ...]]):
        self.executemany_calls.append((sql, list(rows)))

    def close(self):
        return None


class FakeRawConnection:
    def __init__(self) -> None:
        self.cursor_instance = FakeCursor()
        self.commit_calls = 0
        self.rollback_calls = 0
        self.closed = False

    def cursor(self):
        return self.cursor_instance

    def commit(self):
        self.commit_calls += 1

    def rollback(self):
        self.rollback_calls += 1

    def close(self):
        self.closed = True


class FakeEngine:
    def __init__(self) -> None:
        self.raw = FakeRawConnection()

    def raw_connection(self):
        return self.raw


def test_load_synthea_data_batches_rows_into_created_table(tmp_path: Path) -> None:
    patients_file = tmp_path / "patients.csv"
    write_csv(
        patients_file,
        """
        Id,BIRTHDATE,LAST_UPDATE,FIRST
        b9c610cd-28a6-4636-ccb6-c7a0d2a4cb85,2019-02-17,2019-02-17T05:07:38Z,Damon
        c1f1fcaa-82fd-d5b7-3544-c8f9708b06a8,2005-07-04,2019-03-24T05:07:38Z,Thi
        """,
    )

    fake_engine = FakeEngine()
    state = load_synthea_data(
        {
            "engine": fake_engine,
            "batch_size": 2,
            "table_plans": [
                TablePlan(
                    source_file=patients_file,
                    table_name="patients",
                    delimiter=",",
                    columns=(
                        ColumnPlan("Id", "UNIQUEIDENTIFIER", "uuid"),
                        ColumnPlan("BIRTHDATE", "DATE", "date"),
                        ColumnPlan("LAST_UPDATE", "DATETIME2", "datetime"),
                        ColumnPlan("FIRST", "NVARCHAR(MAX)", "text"),
                    ),
                )
            ],
        }
    )

    results = state["execution_results"]
    assert results[0].rows_loaded == 2
    assert fake_engine.raw.cursor_instance.fast_executemany is True
    assert fake_engine.raw.cursor_instance.executed_sql == []
    insert_sql = fake_engine.raw.cursor_instance.executemany_calls[0][0]
    assert insert_sql.startswith("INSERT INTO [synthea].[patients]")
    inserted_rows = fake_engine.raw.cursor_instance.executemany_calls[0][1]
    assert inserted_rows[0][1] == date(2019, 2, 17)
    assert inserted_rows[0][2] == datetime(2019, 2, 17, 5, 7, 38)


def test_load_synthea_data_can_insert_into_prefixed_dbo_table(tmp_path: Path) -> None:
    patients_file = tmp_path / "patients.csv"
    write_csv(
        patients_file,
        """
        Id,FIRST
        b9c610cd-28a6-4636-ccb6-c7a0d2a4cb85,Damon
        """,
    )

    fake_engine = FakeEngine()
    state = load_synthea_data(
        {
            "engine": fake_engine,
            "target_database": "AI_OMOP",
            "schema": "dbo",
            "batch_size": 1,
            "table_plans": [
                TablePlan(
                    source_file=patients_file,
                    table_name="synthea_patients",
                    delimiter=",",
                    columns=(
                        ColumnPlan("Id", "UNIQUEIDENTIFIER", "uuid"),
                        ColumnPlan("FIRST", "NVARCHAR(MAX)", "text"),
                    ),
                )
            ],
        }
    )

    assert state["execution_results"][0].table_name == "synthea_patients"
    assert fake_engine.raw.cursor_instance.executed_sql == ["USE [AI_OMOP]"]
    insert_sql = fake_engine.raw.cursor_instance.executemany_calls[0][0]
    assert insert_sql == "INSERT INTO [dbo].[synthea_patients] ([Id], [FIRST]) VALUES (?, ?)"
