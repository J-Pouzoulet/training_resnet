'''from dotenv import load_dotenv
import os

load_dotenv()

# Get the environment variables
PGHOST = os.getenv("PGHOST")
PGUSER = os.getenv("PGUSER")
PGPORT = os.getenv("PGPORT")
PGDATABASE = os.getenv("PGDATABASE")
PGPASSWORD = os.getenv("PGPASSWORD")
# if using azure blob storage (i.e. your connection string)
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_BLOB_STORAGE_CONNECTION_STRING")
STORAGE_ACCOUNT_NAME = os.getenv("STORAGE_ACCOUNT_NAME")

keys = {'PGHOST', 'PGUSER', 'PGPORT', 'PGDATABASE', 'PGPASSWORD', 'AZURE_STORAGE_CONNECTION_STRING'}   


for key in keys:
    print(f"{key}: {os.getenv(key)}") 
    #os.system(f'export {key}="{os.getenv(key)}"')
    
os.system(f"mlflow server --backend-store-uri postgresql://{PGUSER}:{PGPASSWORD}@{PGHOST}:{PGPORT}/{PGDATABASE} --default-artifact-root wasbs://mlflow-artifacts@{STORAGE_ACCOUNT_NAME}.blob.core.windows.net")
'''
import os
from dotenv import load_dotenv
import subprocess

load_dotenv()

'''def set_env_var(keys):
    # We check if all necessary variables are set
    for key in keys:
        value = os.getenv(key)
        if value is not None:
            pass
        else:
            raise ValueError(f"Environment variable {key} is not set")
    
    # We construct the command to set the environment variables in the current shell
    for key in keys:
        command = f'export {key}="{os.getenv(key)}"'
        print(f"Executing command: {command}")
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
            raise'''


def run_mlflow_server():
    # We load required environment variables
    pguser = os.getenv("PGUSER")
    pgpassword = os.getenv("PGPASSWORD")
    pghost = os.getenv("PGHOST")
    pgport = os.getenv("PGPORT", "5432")  # Default PostgreSQL port
    pgdatabase = os.getenv("PGDATABASE")
    storage_account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
    container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME")
    azure_storage_connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    
    # We check if all necessary variables are set
    if not all([pguser, pgpassword, pghost, pgport, pgdatabase, storage_account_name, container_name]):
        raise EnvironmentError("One or more required environment variables are missing.")

    # We construct the command
    backend_store_uri = f"postgresql://{pguser}:{pgpassword}@{pghost}:{pgport}/{pgdatabase}"
    default_artifact_root = f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net"

    env = os.environ.copy()
    env["AZURE_STORAGE_CONNECTION_STRING"] = azure_storage_connection_string

    command = [
        "mlflow", "server",
        "--backend-store-uri", backend_store_uri,
        "--default-artifact-root", default_artifact_root
    ]
    
    # Add host and port options to the command
    command.extend(["--host", "127.0.0.1", "--port", "5000"])
    
    print(f"Executing command: {' '.join(command)}")

    # Run the command
    try:
        subprocess.run(command, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    keys = ["AZURE_STORAGE_CONNECTION_STRING"]
    try:
        #set_env_var(keys)
        run_mlflow_server()
    except Exception as e:
        print(f"Failed to start MLflow server: {e}")
