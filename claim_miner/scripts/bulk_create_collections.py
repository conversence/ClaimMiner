from typing import List, Tuple
from csv import reader
import argparse

import requests
from claim_miner.pyd_models import (
    CollectionModel,
    StatementModel,
    embedding_model,
    fragment_type,
)
from fastapi.encoders import jsonable_encoder


def create_collections(
    collections: List[Tuple[str, str]], base_url: str, token: str, onnx=True
):
    headers = dict(Authorization=f"Bearer {token}")
    coll_params = dict()
    if onnx:
        coll_params["embeddings"] = [embedding_model.all_minilm_l6_v2.name]
    for coll_name, base_question in collections:
        collection = CollectionModel(name=coll_name, params=coll_params)
        result = requests.post(
            f"{base_url}/api/c", json=jsonable_encoder(collection), headers=headers
        )
        if not result.ok:
            print("Could not create collection {coll_name}")
            continue
        statement = StatementModel(
            text=base_question, language="en", scale=fragment_type.standalone_question
        )
        result = requests.post(
            f"{base_url}/api/c/{coll_name}/statement",
            json=jsonable_encoder(statement),
            headers=headers,
        )
        if not result.ok:
            print(
                "Could not create statement for collection {coll_name}\nStatement: {base_question}"
            )
            continue
        statement_r = result.json()
        theme_id = statement_r["id"]
        result = requests.patch(
            f"{base_url}/api/c/{coll_name}",
            headers=headers,
            json=dict(params=coll_params | dict(theme_id=theme_id)),
        )
        if not result.ok:
            print("Could not set the theme of collection {coll_name} to {theme_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", "-u")
    parser.add_argument("--password", "-p")
    parser.add_argument("--url", default="https://claimminer.conversence.com")
    parser.add_argument("file", type=argparse.FileType(encoding="UTF-8"))
    args = parser.parse_args()
    token_result = requests.post(
        f"{args.url}/api/token", data=dict(username=args.user, password=args.password)
    )
    if not token_result.ok:
        print("Bad user/password")
        exit(1)
    token = token_result.json()["access_token"]
    print(token_result.json())
    reader = reader(args.file)
    rows = list(reader)
    create_collections(rows, args.url, token)
