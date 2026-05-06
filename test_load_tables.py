from pathlib import Path

from src.load_tables import (
    ExecutionResult,
    PlannedStatement,
    execute_sql_plan,
    run_load_tables,
)


def write_sql(path: Path, content: str) -> None:
    path.write_text(content.strip() + "\n", encoding="utf-8")


def test_run_load_tables_plans_files_in_omop_order(tmp_path: Path) -> None:
    write_sql(
        tmp_path / "OMOPCDM_sql_server_5.4_indices.sql",
        "CREATE INDEX idx_person_gender ON @cdmDatabaseSchema.person (gender_concept_id ASC);",
    )
    write_sql(
        tmp_path / "OMOPCDM_sql_server_5.4_constraints.sql",
        """
        ALTER TABLE @cdmDatabaseSchema.person
        ADD CONSTRAINT fpk_person_gender
        FOREIGN KEY (gender_concept_id) REFERENCES @cdmDatabaseSchema.concept (concept_id);
        """,
    )
    write_sql(
        tmp_path / "OMOPCDM_sql_server_5.4_ddl.sql",
        """
        -- table comment
        CREATE TABLE @cdmDatabaseSchema.person (
            person_id integer NOT NULL
        );
        """,
    )
    write_sql(
        tmp_path / "OMOPCDM_sql_server_5.4_primary_keys.sql",
        "ALTER TABLE @cdmDatabaseSchema.person ADD CONSTRAINT xpk_person PRIMARY KEY NONCLUSTERED (person_id);",
    )

    state = run_load_tables(
        ddl_directory=tmp_path,
        target_database="AI_OMOP",
        schema="omop",
        execute=False,
    )

    planned_statements = state["planned_statements"]
    assert [statement.phase for statement in planned_statements] == [
        "ddl",
        "primary_keys",
        "constraints",
        "indices",
    ]
    assert planned_statements[0].statement.startswith("CREATE TABLE [omop].[person]")
    assert planned_statements[1].object_name == "xpk_person"


def test_run_load_tables_can_skip_index_phase(tmp_path: Path) -> None:
    write_sql(
        tmp_path / "OMOPCDM_sql_server_5.4_ddl.sql",
        "CREATE TABLE @cdmDatabaseSchema.person (person_id integer NOT NULL);",
    )
    write_sql(
        tmp_path / "OMOPCDM_sql_server_5.4_indices.sql",
        "CREATE INDEX idx_person_id ON @cdmDatabaseSchema.person (person_id ASC);",
    )

    state = run_load_tables(
        ddl_directory=tmp_path,
        include_indices=False,
        execute=False,
    )

    assert [statement.phase for statement in state["planned_statements"]] == ["ddl"]


class FakeResult:
    def __init__(self, value):
        self.value = value

    def scalar(self):
        return self.value


class FakeConnection:
    def __init__(self) -> None:
        self.executed_sql: list[str] = []
        self.existing_tables = {("dbo", "person")}
        self.existing_constraints = {"xpk_person"}
        self.existing_indexes = {("idx_person_id", "[dbo].[person]")}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, statement, params=None):
        sql_text = str(statement).strip()

        if "FROM sys.tables AS tables" in sql_text:
            key = (params["schema_name"], params["table_name"])
            return FakeResult(1 if key in self.existing_tables else None)

        if "FROM sys.objects" in sql_text:
            return FakeResult(1 if params["object_name"] in self.existing_constraints else None)

        if "FROM sys.indexes" in sql_text:
            key = (params["object_name"], params["table_name"])
            return FakeResult(1 if key in self.existing_indexes else None)

        self.executed_sql.append(sql_text)
        return FakeResult(None)


class FakeEngine:
    def __init__(self) -> None:
        self.connection = FakeConnection()

    def connect(self):
        return self.connection


def test_execute_sql_plan_skips_existing_objects_and_runs_missing_ones() -> None:
    planned_statements = [
        PlannedStatement(
            phase="ddl",
            statement="CREATE TABLE [dbo].[person] (person_id integer NOT NULL)",
            object_type="table",
            object_name="[dbo].[person]",
            table_name="[dbo].[person]",
            source_file="ddl.sql",
        ),
        PlannedStatement(
            phase="primary_keys",
            statement="ALTER TABLE [dbo].[person] ADD CONSTRAINT xpk_person PRIMARY KEY (person_id)",
            object_type="constraint",
            object_name="xpk_person",
            table_name="[dbo].[person]",
            source_file="primary_keys.sql",
        ),
        PlannedStatement(
            phase="constraints",
            statement="ALTER TABLE [dbo].[visit_occurrence] ADD CONSTRAINT fpk_visit_person FOREIGN KEY (person_id) REFERENCES [dbo].[person] (person_id)",
            object_type="constraint",
            object_name="fpk_visit_person",
            table_name="[dbo].[visit_occurrence]",
            source_file="constraints.sql",
        ),
        PlannedStatement(
            phase="indices",
            statement="CREATE INDEX idx_person_id ON [dbo].[person] (person_id ASC)",
            object_type="index",
            object_name="idx_person_id",
            table_name="[dbo].[person]",
            source_file="indices.sql",
        ),
    ]
    fake_engine = FakeEngine()

    state = execute_sql_plan(
        {
            "planned_statements": planned_statements,
            "target_database": "AI_OMOP",
            "engine": fake_engine,
        }
    )

    results = state["execution_results"]
    assert results == [
        ExecutionResult(
            phase="ddl",
            object_type="table",
            object_name="[dbo].[person]",
            source_file="ddl.sql",
            status="skipped",
            reason="already exists",
        ),
        ExecutionResult(
            phase="primary_keys",
            object_type="constraint",
            object_name="xpk_person",
            source_file="primary_keys.sql",
            status="skipped",
            reason="already exists",
        ),
        ExecutionResult(
            phase="constraints",
            object_type="constraint",
            object_name="fpk_visit_person",
            source_file="constraints.sql",
            status="executed",
            reason="created",
        ),
        ExecutionResult(
            phase="indices",
            object_type="index",
            object_name="idx_person_id",
            source_file="indices.sql",
            status="skipped",
            reason="already exists",
        ),
    ]
    assert fake_engine.connection.executed_sql[0] == "USE [AI_OMOP]"
    assert planned_statements[2].statement in fake_engine.connection.executed_sql
