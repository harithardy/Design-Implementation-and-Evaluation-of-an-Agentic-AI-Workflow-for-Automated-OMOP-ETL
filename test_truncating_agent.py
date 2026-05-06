from pathlib import Path

from src.load_tables import PlannedStatement
from src.truncating_agent import TruncationResult, empty_omop_tables, run_truncating_agent


def write_sql(path: Path, content: str) -> None:
    path.write_text(content.strip() + "\n", encoding="utf-8")


def test_run_truncating_agent_plans_only_omop_tables_from_ddl_files(tmp_path: Path) -> None:
    write_sql(
        tmp_path / "OMOPCDM_sql_server_5.4_ddl.sql",
        """
        CREATE TABLE @cdmDatabaseSchema.person (person_id integer NOT NULL);
        CREATE TABLE @cdmDatabaseSchema.visit_occurrence (visit_occurrence_id integer NOT NULL);
        """,
    )
    write_sql(
        tmp_path / "synthea_tables.sql",
        "CREATE TABLE @cdmDatabaseSchema.synthea_patients (id integer NOT NULL);",
    )
    write_sql(
        tmp_path / "OMOPCDM_sql_server_5.4_constraints.sql",
        """
        ALTER TABLE @cdmDatabaseSchema.visit_occurrence
        ADD CONSTRAINT fpk_visit_person
        FOREIGN KEY (person_id) REFERENCES @cdmDatabaseSchema.person (person_id);
        """,
    )

    state = run_truncating_agent(
        ddl_directory=tmp_path,
        target_database="AI_OMOP",
        schema="dbo",
        execute=False,
    )

    assert [statement.table_name for statement in state["planned_tables"]] == [
        "[dbo].[person]",
        "[dbo].[visit_occurrence]",
    ]
    assert all(
        "synthea_patients" not in table_name for table_name in state["summary"]["planned_table_names"]
    )


class FakeResult:
    def __init__(self, value):
        self.value = value

    def scalar(self):
        return self.value


class FakeConnection:
    def __init__(self) -> None:
        self.executed_sql: list[str] = []
        self.existing_tables = {
            ("dbo", "person"),
            ("dbo", "visit_occurrence"),
            ("dbo", "synthea_patients"),
        }
        self.row_counts = {
            "[dbo].[person]": 3,
            "[dbo].[visit_occurrence]": 2,
            "[dbo].[synthea_patients]": 7,
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, statement, params=None):
        sql_text = str(statement).strip()

        if "FROM sys.tables AS tables" in sql_text:
            key = (params["schema_name"], params["table_name"])
            return FakeResult(1 if key in self.existing_tables else None)

        if sql_text.startswith("SELECT COUNT_BIG(1) AS row_count FROM "):
            table_name = sql_text.removeprefix("SELECT COUNT_BIG(1) AS row_count FROM ")
            return FakeResult(self.row_counts[table_name])

        if sql_text.startswith("DELETE FROM "):
            table_name = sql_text.removeprefix("DELETE FROM ")
            self.row_counts[table_name] = 0

        self.executed_sql.append(sql_text)
        return FakeResult(None)


class FakeEngine:
    def __init__(self) -> None:
        self.connection = FakeConnection()

    def connect(self):
        return self.connection


def test_empty_omop_tables_clears_only_planned_ddl_tables() -> None:
    fake_engine = FakeEngine()
    state = empty_omop_tables(
        {
            "engine": fake_engine,
            "target_database": "AI_OMOP",
            "planned_tables": [
                PlannedStatement(
                    phase="ddl",
                    statement="CREATE TABLE [dbo].[person] (person_id integer NOT NULL)",
                    object_type="table",
                    object_name="[dbo].[person]",
                    table_name="[dbo].[person]",
                    source_file="OMOPCDM_sql_server_5.4_ddl.sql",
                ),
                PlannedStatement(
                    phase="ddl",
                    statement="CREATE TABLE [dbo].[visit_occurrence] (visit_occurrence_id integer NOT NULL)",
                    object_type="table",
                    object_name="[dbo].[visit_occurrence]",
                    table_name="[dbo].[visit_occurrence]",
                    source_file="OMOPCDM_sql_server_5.4_ddl.sql",
                ),
                PlannedStatement(
                    phase="ddl",
                    statement="CREATE TABLE [dbo].[drug_exposure] (drug_exposure_id integer NOT NULL)",
                    object_type="table",
                    object_name="[dbo].[drug_exposure]",
                    table_name="[dbo].[drug_exposure]",
                    source_file="OMOPCDM_sql_server_5.4_ddl.sql",
                ),
            ],
        }
    )

    assert state["execution_results"] == [
        TruncationResult(
            table_name="[dbo].[drug_exposure]",
            status="skipped",
            rows_before=0,
            rows_after=0,
            reason="table not found",
        ),
        TruncationResult(
            table_name="[dbo].[person]",
            status="emptied",
            rows_before=3,
            rows_after=0,
            reason="table cleared",
        ),
        TruncationResult(
            table_name="[dbo].[visit_occurrence]",
            status="emptied",
            rows_before=2,
            rows_after=0,
            reason="table cleared",
        ),
    ]
    assert fake_engine.connection.row_counts["[dbo].[synthea_patients]"] == 7
    assert all("synthea_patients" not in sql for sql in fake_engine.connection.executed_sql)
    assert fake_engine.connection.executed_sql == [
        "USE [AI_OMOP]",
        "ALTER TABLE [dbo].[person] NOCHECK CONSTRAINT ALL",
        "ALTER TABLE [dbo].[visit_occurrence] NOCHECK CONSTRAINT ALL",
        "DELETE FROM [dbo].[person]",
        "DELETE FROM [dbo].[visit_occurrence]",
        "ALTER TABLE [dbo].[visit_occurrence] WITH CHECK CHECK CONSTRAINT ALL",
        "ALTER TABLE [dbo].[person] WITH CHECK CHECK CONSTRAINT ALL",
    ]
