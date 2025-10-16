import dotenv
from notion_client import Client
dotenv.load_dotenv()
import os

client = Client(auth=os.environ["NOTION_TOKEN"])
db_id = os.environ["NOTION_DATABASE_ID"]

db = client.databases.retrieve(database_id=db_id)
title_prop = next((n for n,m in db["properties"].items() if m.get("type")=="title"), "Name")
resp = client.databases.query(database_id=db_id, page_size=5)
print(title_prop, [ "".join(t.get("plain_text","") for t in p.get(title_prop,{}).get("title",[])) for p in [r["properties"] for r in resp["results"]] ])