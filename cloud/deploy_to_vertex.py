"""Deploy training job to Vertex AI."""
import argparse
from google.cloud import aiplatform
import yaml

def deploy_training_job(
    project_id: str,
    location: str,
    job_name: str,
    config_path: str
):
    """Deploy a training job to Vertex AI."""
    # Initialize Vertex AI with staging bucket
    aiplatform.init(
        project=project_id,
        location=location,
        staging_bucket="gs://airotechbkt"  # Using your existing bucket
    )
    
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Create custom training job
    worker_pool_specs = [{
        "machine_spec": {
            "machine_type": config["workerPoolSpecs"]["machineSpec"]["machineType"],
            "accelerator_type": config["workerPoolSpecs"]["machineSpec"]["acceleratorType"],
            "accelerator_count": config["workerPoolSpecs"]["machineSpec"]["acceleratorCount"],
        },
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
    job.run(
        sync=True,
        enable_web_access=True,
        service_account=config["serviceAccount"]
    )

def main():
    parser = argparse.ArgumentParser(description='Deploy training job to Vertex AI')
    parser.add_argument('--project', required=True, help='Google Cloud project ID')
    parser.add_argument('--location', default='us-central1', help='Google Cloud region')
    parser.add_argument('--job-name', default='rohingya-translator-training',
                      help='Name for the training job')
    parser.add_argument('--config', default='cloud/vertex_ai_config.yaml',
                      help='Path to Vertex AI configuration file')
    args = parser.parse_args()
    
    deploy_training_job(
        project_id=args.project,
        location=args.location,
        job_name=args.job_name,
        config_path=args.config
    )

if __name__ == '__main__':
    main()
