"""Deploy training job to Vertex AI."""
import argparse
from google.cloud import aiplatform
from google.cloud import storage
import yaml
import os
from pathlib import Path

def create_bucket_if_not_exists(bucket_name: str, project_id: str, location: str):
    """Create a new GCS bucket if it doesn't exist."""
    storage_client = storage.Client(project=project_id)
    
    try:
        bucket = storage_client.get_bucket(bucket_name)
        print(f"Bucket {bucket_name} already exists")
        return bucket
    except Exception:
        print(f"Creating new bucket {bucket_name} in {location}")
        bucket = storage_client.create_bucket(
            bucket_name,
            location=location,
        )
        return bucket

def upload_directory_to_gcs(local_path: str, bucket_name: str, gcs_path: str, project_id: str):
    """Upload a local directory to GCS."""
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    
    local_path = Path(local_path)
    if not local_path.exists():
        print(f"Warning: Local path {local_path} does not exist, skipping upload")
        return
    
    for local_file in local_path.rglob("*"):
        if local_file.is_file():
            relative_path = local_file.relative_to(local_path)
            blob_path = f"{gcs_path}/{relative_path}".replace("\\", "/")
            blob = bucket.blob(blob_path)
            
            print(f"Uploading {local_file} to gs://{bucket_name}/{blob_path}")
            blob.upload_from_filename(str(local_file))

def setup_gcs_directories(project_id: str, bucket_name: str):
    """Set up necessary GCS directories and upload initial data."""

    
    # Upload data if it exists
    upload_directory_to_gcs(
        "data",
        bucket_name,
        "data",
        project_id
    )
    
    # Create empty directories with .gitkeep files
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    
    # Create model registry directory (empty)
    bucket.blob("model-registry/.gitkeep").upload_from_string("")
    
    # Create models directory (empty)
    bucket.blob("models/.gitkeep").upload_from_string("")

def deploy_training_job(
    project_id: str,
    location: str,
    job_name: str,
    config_path: str
):
    """Deploy a training job to Vertex AI."""
    # Create a regional bucket for Vertex AI
    bucket_name = "rohingya-translator-vertex"
    create_bucket_if_not_exists(bucket_name, project_id, location)
    staging_bucket = f"gs://{bucket_name}"
    
    # Set up GCS directories and upload initial data
    setup_gcs_directories(project_id, bucket_name)
    
    # Initialize Vertex AI with staging bucket
    aiplatform.init(
        project=project_id,
        location=location,
        staging_bucket=staging_bucket
    )
    
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Create custom training job
    job = aiplatform.CustomJob(
        display_name=job_name,
        worker_pool_specs=[*config["workerPoolSpecs"]],
        staging_bucket=staging_bucket,
    )
    
    # Run the training job
    job.run(
        service_account=config["serviceAccount"],
        network=config.get("network"),
        timeout=config.get("scheduling", {}).get("timeout")
    )
    
    print(f"Job {job_name} launched successfully")
    print(f"View the job at: https://console.cloud.google.com/vertex-ai/locations/{location}/training/{job.resource_name}")
    return job

def main():
    parser = argparse.ArgumentParser(description="Deploy training job to Vertex AI")
    parser.add_argument("--project-id", required=True, help="Google Cloud project ID")
    parser.add_argument("--location", default="us-central1", help="Region for the training job")
    parser.add_argument("--job-name", default="rohingya-translator-training", help="Display name for the job")
    parser.add_argument("--config", default="src/cloud/vertex_ai_config.yaml", help="Path to job config YAML")
    
    args = parser.parse_args()
    deploy_training_job(args.project_id, args.location, args.job_name, args.config)

if __name__ == '__main__':
    main()
