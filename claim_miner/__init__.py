"""
ClaimMiner main package.
Set up database access, load ORM models, HashFS access, and a few common utility functions.
"""
# Copyright Society Library and Conversence 2022-2024

import asyncio
from itertools import chain
import os
from pathlib import Path
from configparser import ConfigParser
import logging
import logging.config

import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy import select
from hashfs import HashFS

from .dispatcher import Dispatcher
from .utils import as_bool, run_sync

hashfs = HashFS("files", depth=4, width=1, algorithm="sha256")

config = ConfigParser()
# print(Path(__file__).parent.joinpath("config.ini"))
config.read(Path(__file__).parent.parent.joinpath("config.ini"))
production = as_bool(os.environ.get("PRODUCTION", False))
is_testing = os.environ.get("PYTEST_VERSION", None) is not None
target_db = os.environ.get(
    "TARGET_DB",
    ("test" if is_testing else ("production" if production else "development")),
)
openai_key = config.get("openai", "api_key", fallback="")
openai_org = config.get("openai", "organization", fallback="")
anthropic_key = config.get("anthropic", "api_key", fallback="")
os.environ["OPENAI_API_KEY"] = openai_key
os.environ["OPENAI_ORGANIZATION"] = openai_org
os.environ["CLIENT_TYPE"] = "CLAIM_MINER"

pool_size = int(os.environ.get("SQLALCHEMY_POOL_SIZE", 5))


def set_log_config(section: str):
    if section:
        path = Path(f"{section}.ini")
    if path.exists():
        logging.config.fileConfig(path)
    else:
        logging.basicConfig()


if is_testing:
    from test.mock_dispatcher import MockDispatcher

    dispatcher: Dispatcher = MockDispatcher()
else:
    # TODO: Make this dynamic and config-sensitive
    from .kafka import KafkaDispatcher

    dispatcher: Dispatcher = KafkaDispatcher()

_db_config_get_nonce = object()


def db_config_get(key: str, default=_db_config_get_nonce):
    if default is _db_config_get_nonce or config.has_option(target_db, key):
        return config.get(target_db, key)
    return default


db_name = db_config_get("database")

base_connection_string = f"{db_config_get('owner')}:{db_config_get('owner_password')}@{config.get('postgres', 'host')}:{config.get('postgres', 'port')}/{db_config_get('database')}"
sync_connection_string = f"postgresql://{base_connection_string}"
sync_connection_string_sqla = f"postgresql+psycopg2://{base_connection_string}"
async_connection_string = f"postgresql+asyncpg://{base_connection_string}"
os.environ["POSTGRES_CONNECTION_STRING"] = sync_connection_string

engine = create_async_engine(async_connection_string, pool_size=pool_size)
sync_maker = sessionmaker(expire_on_commit=False)
Session = sessionmaker(
    bind=engine,
    expire_on_commit=False,
    class_=AsyncSession,
    sync_session_class=sync_maker,
)


def escape_fn(s):
    return sa.String("").literal_processor(dialect=engine.dialect)(value=s)
