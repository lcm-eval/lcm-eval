import os
import re
import time
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

from cross_db_benchmark.benchmark_tools.database import DatabaseConnection, ExecutionMode
from cross_db_benchmark.benchmark_tools.utils import load_schema_sql, load_schema_json, load_column_statistics
from cross_db_benchmark.meta_tools.scale_dataset import extract_scale_columns, find_numeric_offset, extract_type


class PostgresDatabaseConnection(DatabaseConnection):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_database(self, dataset, data_dir, force=False):
        # drop and create database
        exists_res = self.check_if_database_exists()

        if not force and exists_res:
            print(f"Skipping loading of {self.db_name} since it already exists")
            return

        self.submit_query(f'DROP DATABASE IF EXISTS {self.db_name};', db_created=False)
        self.submit_query(f'CREATE DATABASE {self.db_name};', db_created=False)

        schema_sql = load_schema_sql(dataset, 'postgres.sql')
        self.submit_query(schema_sql)
        schema = load_schema_json(dataset)

        print(f"Loading tables")
        for t in schema.tables:
            start_t = time.perf_counter()
            table_path = os.path.join(data_dir, f'{t}.csv')
            table_path = Path(table_path).resolve()
            load_cmd = f"COPY \"{t}\" FROM '{table_path}' {schema.db_load_kwargs.postgres};"
            self.submit_query(load_cmd)
            print(f"Loaded {t} in {time.perf_counter() - start_t:.2f} secs")

        print("Starting vacuum analyze...")
        self.submit_query("VACUUM ANALYZE;")

    def replicate_tuples(self, dataset, data_dir, no_prev_replications, vac_analyze=True):
        # deactivate statement timeout during this
        self.set_statement_timeout(0, verbose=True)

        # adapt foreign keys
        schema = load_schema_json(dataset)
        pg_schema, _, scale_columns = extract_scale_columns(schema, dataset)
        column_stats = {k: vars(v) for k, v in vars(load_column_statistics(dataset)).items()}
        custom_nan_values = ['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND',
                             '1.#QNAN', '<NA>', 'N/A', 'NULL', 'NaN', 'n/a', 'nan', 'null']
        read_kwargs = dict(**vars(schema.csv_kwargs), keep_default_na=False, na_values=custom_nan_values)

        for t in schema.tables:
            # find columns in the right order
            columns = pd.read_csv(os.path.join(data_dir, f'{t}.csv'), nrows=1, **read_kwargs).columns

            project_rows = []

            for c in columns:
                if c in scale_columns[t]:
                    type = extract_type(pg_schema, t, c).lower()
                    if type.startswith('varchar'):
                        projection = f'\"{c}\" || \'R\' AS \"{c}\"'
                    elif type.startswith('int'):
                        offset = find_numeric_offset(c, column_stats, schema, t)
                        offset *= 2 ** no_prev_replications
                        projection = f'\"{c}\"+{offset} AS \"{c}\"'
                    else:
                        raise NotImplementedError(type)
                else:
                    projection = f'\"{c}\" AS \"{c}\"'

                assert projection is not None
                project_rows.append(projection)

            replicate_sql = f"INSERT INTO \"{t}\" SELECT {', '.join(project_rows)} FROM \"{t}\";"

            start_t = time.perf_counter()
            self.submit_query(replicate_sql)
            print(replicate_sql)
            print(f"Replicated {t} in {time.perf_counter() - start_t:.2f} secs")

        if vac_analyze:
            start_t = time.perf_counter()
            print('Vacuum Analyze...')
            self.submit_query('VACUUM ANALYZE')
            print(f"Ran vacuum analyze in {time.perf_counter() - start_t:.2f} secs")

    def check_if_database_exists(self):
        exists_res = self.get_result(f"""SELECT EXISTS (
         SELECT datname FROM pg_catalog.pg_database WHERE datname = '{self.db_name}'
        );""", db_created=False)
        return exists_res[0][0]

    def remove_remaining_fk_indexes(self):
        benchmark_idx_query = """ SELECT indexname FROM pg_indexes 
            WHERE schemaname = 'public' AND indexname LIKE 'zero_shot_%'
        """
        index_rows = self.get_result(benchmark_idx_query)
        for r in index_rows:
            index_name = r[0]
            print(f"Dropping previously created index {index_name}")
            self.drop_index(index_name)

    def drop_index(self, index_name):
        self.submit_query(f'DROP INDEX "{index_name}";')

    def create_index(self, table, column):
        index_name = f"zero_shot_{table}_{column}"
        self.submit_query(f'CREATE INDEX "{index_name}" ON "{table}" ("{column}");')
        return index_name

    def create_db(self):
        exists_res = self.submit_query(f"CREATE DATABASE {self.db_name};", db_created=False)
        return exists_res

    def test_join_conditions(self, dataset):
        schema = load_schema_json(dataset)

        for table_left, cols_left, table_right, cols_right in schema.relationships:
            if not (isinstance(cols_left, tuple) or isinstance(cols_left, list)):
                cols_left = [cols_left]
                cols_right = [cols_right]

            join_conds = ' AND '.join([f'"{table_left}"."{c_left}" = "{table_right}"."{c_right}"'
                                       for c_left, c_right in zip(cols_left, cols_right)])

            res = self.get_result(f"SELECT COUNT(*) FROM \"{table_left}\" JOIN \"{table_right}\" ON {join_conds};")
            card = res[0][0]
            print(f"{join_conds}: {card} join tuples")
            if not card > 1:
                print("WARNING: low cardinality. Check join condition")

    def set_statement_timeout(self, timeout_sec, verbose=True):
        self.submit_query(f"ALTER DATABASE {self.db_name} SET statement_timeout = {timeout_sec * 1000};",
                          db_created=False)
        if verbose:
            print(f"Set timeout to {timeout_sec} secs for {self.db_name}.")

    def run_query_collect_statistics(self,
                                     sql: str,
                                     mode: ExecutionMode,
                                     repetitions: int = 3,
                                     prefix: str = "",
                                     explain_only: bool = False,
                                     include_hint_notices: bool = True,
                                     hint_validation: bool = True) -> dict:
        analyze_plans = None
        verbose_plan = None
        hint_notices = None
        timeout = False

        if mode == ExecutionMode.JSON_OUTPUT:
            explain_only_query = "(analyze true, format json)"
            verbose_query = "(verbose true, analyze true, format json)"

        elif mode == ExecutionMode.RAW_OUTPUT:
            explain_only_query = "analyze"
            verbose_query = "verbose"
        else:
            raise Exception("Execution mode not supported")

        try:
            verbose_plan = self.get_result(f"{prefix} explain {verbose_query} {sql}",
                                           include_hint_notices=include_hint_notices, mode=mode)

            analyze_plans = []
            if not explain_only:
                for i in range(repetitions):
                    statement = f"{prefix} explain {explain_only_query} {sql}"
                    curr_analyze_plan = self.get_result(statement, include_hint_notices=include_hint_notices, mode=mode)
                    analyze_plans.append(curr_analyze_plan)

        except psycopg2.errors.QueryCanceled as e:
            timeout = True
            print(f"Hit the timeout for query: {sql}")

        except psycopg2.errors.UndefinedFunction:
            # ToDo: Not fully sure about how Benjamin handled this?
            print(f"No operator matches the given name and argument types for {sql}")

        except psycopg2.errors.SyntaxError:
            print("Invalid due to Syntax error")

        except psycopg2.errors.InvalidTextRepresentation:
            print("Invalid due to Invalid text error")

        if hint_notices and hint_validation:
            self.validate_hint_execution(hint_notices, sql)

        return dict(analyze_plans=analyze_plans,
                    verbose_plan=verbose_plan,
                    timeout=timeout,
                    hint_notices=hint_notices)

    def validate_hint_execution(self, hint_notices: str, sql: str) -> None:
        original_hint = sql.split("*/")[0].strip(" ").strip("/*+ ")

        # Get state dump from hint_notices
        hint_logs = dict()
        state_dump = None
        for notice in hint_notices:
            if "syntax error" in notice:
                raise SyntaxError(f"Syntax error found for hinting {sql}: {notice}")
            if "HintStateDump" in notice:
                state_dump = notice.strip("LOG: pg_hint_plan[qno=0x0]: HintStateDump: ")
                state_dump = re.sub(r"[{}]", "", state_dump)
                for entry in state_dump.split(","):
                    hint_logs[entry.split(":")[0].strip()] = entry.split(":")[1].strip()

        if not state_dump:
            raise RuntimeError("No hint state dump found in hint notices")

        # Check hint logs for correctness (no errors, duplicates, all hints used)
        assert hint_logs["duplicate hints"] == '(none)', f'Found duplicate hints: {hint_logs["duplicate hints"]}'
        assert hint_logs["error hints"] == '(none)', f'Found error hints: {hint_logs["error hints"]}'
        assert hint_logs["not used hints"] == '(none)', f'Found unused hints: {hint_logs["not used hints"]}'

        # Check actual executed hints for equality
        original_hint_list = self.parse_components(original_hint)
        used_hint_list = self.parse_components(hint_logs["used hints"])
        assert original_hint_list == used_hint_list, \
            (f'Target hints not equal to used hints. '
             f'Expected {original_hint_list.difference(used_hint_list)} but found {used_hint_list.difference(original_hint_list)}')
        return

    @staticmethod
    def parse_components(s):
        components = []
        current_component = ''
        depth = 0

        for char in s:
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
            current_component += char
            # Check for the end of a component when the depth is 0
            if depth == 0 and (char == ' ' or char == ')'):
                current_component = current_component.strip()
                if current_component:
                    components.append(current_component.strip())
                    current_component = ''
        return set(components)

    def collect_db_statistics(self):
        # column stats
        stats_query = """
            SELECT s.tablename, s.attname, s.null_frac, s.avg_width, s.n_distinct, s.correlation, c.data_type 
            FROM pg_stats s
            JOIN information_schema.columns c ON s.tablename=c.table_name AND s.attname=c.column_name
            WHERE s.schemaname='public';
        """
        column_stats_names, column_stats_rows = self.get_result(stats_query, include_column_names=True)
        column_stats = self.transform_dicts(column_stats_names, column_stats_rows)

        # table stats
        table_stats_names, table_stats_rows = self.get_result(
            f"SELECT relname, reltuples, relpages from pg_class WHERE relkind = 'r';",
            include_column_names=True)
        table_stats = self.transform_dicts(table_stats_names, table_stats_rows)
        assert column_stats != {} and table_stats !={}, f"No statistics found for {self.db_name}"
        return dict(column_stats=column_stats, table_stats=table_stats)

    def get_result(self, sql: str, include_column_names: bool = False, db_created: bool = True,
                   include_hint_notices: bool = False, mode: ExecutionMode = ExecutionMode.RAW_OUTPUT):
        connection, cursor = self.get_cursor(db_created=db_created)
        cursor.execute(sql)
        records = cursor.fetchall()
        self.close_conn(connection, cursor)
        if include_column_names:
            return [desc[0] for desc in cursor.description], records

        if include_hint_notices:
            assert connection.notices, f"No hint logs found during execution of {sql}"
            return connection.notices, records

        if mode == ExecutionMode.JSON_OUTPUT:
            return records[0][0][0]
        else:
            return records

    def get_cursor(self, db_created=True):
        if db_created:
            connection = psycopg2.connect(database=self.db_name, **self.database_kwargs)
        else:
            connection = psycopg2.connect(**self.database_kwargs)
        connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = connection.cursor()
        return connection, cursor

    def submit_query(self, sql, db_created=True):
        connection, cursor = self.get_cursor(db_created=db_created)
        cursor.execute(sql)
        connection.commit()
        self.close_conn(connection, cursor)

    def close_conn(self, connection, cursor):
        if connection:
            cursor.close()
            connection.close()
