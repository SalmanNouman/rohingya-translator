"""Deploy training job to Vertex AI."""
import argparse
from google.cloud import aiplatform
from google.cloud import storage
import yaml

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
    
    # Initialize Vertex AI with staging bucket
    aiplatform.init(
        project=project_id,
        location=location,
        staging_bucket=staging_bucket
    )
    
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Update GCS paths to use the new bucket
    config["workerPoolSpecs"]["containerSpec"]["args"] = [
        arg.replace("airotechbkt", bucket_name) for arg in config["workerPoolSpecs"]["containerSpec"]["args"]
    ]
    
    # Create custom training job
    machine_spec = {
        "machine_type": config["workerPoolSpecs"]["machineSpec"]["machineType"],
    }
    
    # Add accelerator config if present
    if "acceleratorType" in config["workerPoolSpecs"]["machineSpec"]:
        machine_spec.update({
            "accelerator_type": config["workerPoolSpecs"]["machineSpec"]["acceleratorType"],
            "accelerator_count": config["workerPoolSpecs"]["machineSpec"]["acceleratorCount"],
        })
    
    worker_pool_specs = [{
        "machine_spec": machine_spec,
        "replica_count": config["workerPoolSpecs"]["replicaCount"],
        "container_spec": {
            "image_uri": config["workerPoolSpecs"]["containerSpec"]["imageUri"],
            "args": config["workerPoolSpecs"]["containerSpec"]["args"],
        }
    }]
    
    job = aiplatform.CustomJob(
        display_name=job_name,
        worker_pool_specs=worker_pool_specs,
        labels={"training_job": "rohingya_translator"}
    )
    
    # Start the training
    try:
        job.run(
            sync=True,  # Wait for the job to complete
            restart_job_on_worker_restart=True,
        )
        print(f"Training job {job_name} completed successfully!")
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Deploy training job to Vertex AI")
    parser.add_argument("--project-id", required=True, help="Google Cloud project ID")
    parser.add_argument("--location", default="us-central1", help="Region for the training job")
    parser.add_argument("--job-name", required=True, help="Name for the training job")
    parser.add_argument(
        "--config", 
        default="cloud/vertex_ai_config.yaml",
        help="Path to Vertex AI configuration file"
    )
    
    args = parser.parse_args()
    
    try:
        deploy_training_job(
            project_id=args.project_id,
            location=args.location,
            job_name=args.job_name,
            config_path=args.config
        )
    except Exception as e:
        print(f"Failed to deploy training job: {str(e)}")
        exit(1)

if __name__ == '__main__':
    main()
