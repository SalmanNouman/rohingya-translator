"""Upload data and configuration to Google Cloud Storage."""
import os
from google.cloud import storage
from pathlib import Path
import argparse

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

def upload_directory(bucket_name: str, source_dir: Path, destination_prefix: str, project_id: str, location: str):
    """Upload a directory to GCS bucket."""
    # Ensure bucket exists
    create_bucket_if_not_exists(bucket_name, project_id, location)
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    for local_file in source_dir.rglob('*'):
        if local_file.is_file():
            # Skip local config files
            if 'configs/local' in str(local_file):
                continue
                
            relative_path = local_file.relative_to(source_dir)
            destination_blob_name = f"{destination_prefix}/{relative_path}"
            blob = bucket.blob(destination_blob_name)
            
            print(f"Uploading {local_file} to {destination_blob_name}")
            blob.upload_from_filename(str(local_file))

def verify_gcs_data(bucket_name: str, project_id: str):
    """Verify that all required data and config files are present in GCS."""
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    
    required_files = [
        'configs/model_config.yaml',
        'data/processed/train.en',
        'data/processed/train.roh',
        'data/processed/val.en',
        'data/processed/val.roh',
        'data/processed/test.en',
        'data/processed/test.roh'
    ]
    
    missing_files = []
    for file_path in required_files:
        blob = bucket.blob(file_path)
        try:
            blob.reload()  # Load blob metadata
            print(f"[OK] Found: {file_path} ({blob.size / 1024:.2f} KB)")
        except Exception:
            missing_files.append(file_path)
            print(f"[X] Missing: {file_path}")
    
    if missing_files:
        print("\n[WARNING] Some required files are missing. Please upload them before proceeding.")
        return False
    
    print("\n[SUCCESS] All required files are present in GCS!")
    return True

def main():
    parser = argparse.ArgumentParser(description='Upload data to Google Cloud Storage')
    parser.add_argument('--bucket', required=True, help='GCS bucket name')
    parser.add_argument('--project', required=True, help='Google Cloud project ID')
    parser.add_argument('--location', default='us-central1', help='GCS bucket location')
    parser.add_argument('--verify-only', action='store_true', help='Only verify files without uploading')
    args = parser.parse_args()
    
    # Set Google Cloud project
    os.environ['GOOGLE_CLOUD_PROJECT'] = args.project
    
    if args.verify_only:
        verify_gcs_data(args.bucket, args.project)
        return
        
    # Upload data
    data_dir = Path('data')
    upload_directory(args.bucket, data_dir, 'data', args.project, args.location)
    
    # Upload configurations
    config_dir = Path('configs')
    upload_directory(args.bucket, config_dir, 'configs', args.project, args.location)
    
    print("Upload complete!")

if __name__ == '__main__':
    main()
