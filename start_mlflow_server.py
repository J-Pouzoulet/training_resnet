import os
from dotenv import load_dotenv
import subprocess
import argparse

load_dotenv()

# We set up argument parser so that they can be passed from the command line
parser = argparse.ArgumentParser(description="Start an MLflow server using Sqlite and local folder or Azure backend(Posgresql and Blob Storage).")
parser.add_argument("--storage", type=str, default="Local", help="Local, Azure")

# We parse the arguments
args = parser.parse_args()

# We assign the arguments to variables
storage = args.storage

def run_mlflow_server(storage: str):
    if storage == "Azure":
        # We load required environment variables
        pguser = os.getenv("PGUSER")
        pgpassword = os.getenv("PGPASSWORD")
        pghost = os.getenv("PGHOST")
        pgport = os.getenv("PGPORT", "5432")  # Default PostgreSQL port
        pgdatabase = os.getenv("PGDATABASE")
        
        storage_account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
        container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME")
        sas_token = os.getenv("AZURE_STORAGE_SAS_TOKEN")
        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        
        # We check if all necessary variables are set
        if not all([pguser, pgpassword, pghost, pgport, pgdatabase, storage_account_name, container_name]):
            raise EnvironmentError("One or more required environment variables are missing.")

        if connection_string:
            # Set it in the shell environment
            os.environ["AZURE_STORAGE_CONNECTION_STRING"] = connection_string
            print("Connection string set in the shell environment.")
        
        # We construct the command
        backend_store_uri = f"postgresql://{pguser}:{pgpassword}@{pghost}:{pgport}/{pgdatabase}"
        default_artifact_root = f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net?{sas_token}"

        command = [
            "mlflow", "server",
            "--backend-store-uri", backend_store_uri,
            "--default-artifact-root", default_artifact_root
        ]
        
        # Add host and port options to the command
        command.extend(["--host", "127.0.0.1"])
        
        print(f"Executing command: {' '.join(command)}")

        # Run the command
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
            raise

    else:
        command = [
            "mlflow", "server",
            "--backend-store-uri", "sqlite:///mlflow.db",
            "--default-artifact-root", "file:./mlruns"
        ]
        
        # Add host and port options to the command
        command.extend(["--host", "127.0.0.1"])
            
        print(f"Executing command: {' '.join(command)}")
            
       # Run the command
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
            raise     
            
if __name__ == "__main__":
    try:
        run_mlflow_server(storage)
    except Exception as e:
        print(f"Failed to start MLflow server: {e}")
