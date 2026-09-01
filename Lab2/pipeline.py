import pandas as pd
import sqlite3


def extract(path: str = 'data/raw/source.csv') -> pd.DataFrame:
    df = pd.read_csv(path, encoding='latin1')  # file is Latin-1, not UTF-8
    return df


def transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=['Product Base Margin']).copy()
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

    # derived feature: profit as a % of sales
    df['profit_margin_pct'] = (df['profit'] / df['sales']) * 100

    return df.reset_index(drop=True)


def load(df: pd.DataFrame, db='data/warehouse.db', table='clean_data'):
    with sqlite3.connect(db) as conn:
        df.to_sql(table, conn, if_exists='replace', index=False)
    df.to_parquet('data/clean_data.parquet', index=False)


def validate(df: pd.DataFrame):
    assert len(df) > 0, 'pipeline produced no rows'
    assert not df.isnull().any().any(), 'null values remain'
    assert df.columns.is_unique, 'duplicate column names'
    print(f'OK: {len(df)} rows, {df.shape[1]} columns')


def confirm(db='data/warehouse.db', table='clean_data'):
    con = sqlite3.connect(db)
    print(pd.read_sql(f'SELECT * FROM {table} LIMIT 5', con))


def run(path: str):
    df = extract(path)
    df = transform(df)
    validate(df)
    load(df)
    print('Pipeline complete.')
    confirm()


if __name__ == '__main__':
    run('data/raw/source.csv')