import os
from databricks.sdk import WorkspaceClient
from dotenv import load_dotenv

load_dotenv()

# The user has databricks_token in .env, but WorkspaceClient expects DATABRICKS_TOKEN
os.environ["DATABRICKS_HOST"] = os.environ.get("MLFLOW_URI")
os.environ["DATABRICKS_TOKEN"] = os.environ.get("databricks_token")

w = WorkspaceClient()

print("Listing Catalogs:")
try:
    for catalog in w.catalogs.list():
        print(f"- {catalog.name}")
except Exception as e:
    print(f"Error listing catalogs: {e}")
