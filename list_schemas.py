import os
from databricks.sdk import WorkspaceClient
from dotenv import load_dotenv

load_dotenv()

os.environ["DATABRICKS_HOST"] = os.environ.get("MLFLOW_URI")
os.environ["DATABRICKS_TOKEN"] = os.environ.get("databricks_token")

w = WorkspaceClient()

print("Listing Schemas in 'workspace' catalog:")
try:
    for schema in w.schemas.list(catalog_name="workspace"):
        print(f"- {schema.name}")
except Exception as e:
    print(f"Error listing schemas: {e}")
