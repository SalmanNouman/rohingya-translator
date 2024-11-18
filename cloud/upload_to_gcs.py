"""Upload data and configuration to Google Cloud Storage."""
import os
from google.cloud import storage
from pathlib import Path
import argparse

def upload_directory(bucket_name: str, source_dir: Path, destination_prefix: str):
    """Upload a directory to GCS bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    for local_file in source_dir.rglob('*'):
        if local_file.is_file():
            relative_path = local_file.relative_to(source_dir)
            destination_blob_name = f"{destination_prefix}/{relative_path}"
            blob = bucket.blob(destination_blob_name)
            
            print(f"Uploading {local_file} to {destination_blob_name}")
            blob.upload_from_filename(str(local_file))

def main():
    parser = argparse.ArgumentParser(description='Upload data to Google Cloud Storage')
    parser.add_argument('--bucket', required=True, help='GCS bucket name')
    parser.add_argument('--project', required=True, help='Google Cloud project ID')
    args = parser.parse_args()
    
    # Set Google Cloud project
    os.environ['GOOGLE_CLOUD_PROJECT'] = args.project
    
    # Upload data
    data_dir = Path('data')
    upload_directory(args.bucket, data_dir, 'data')
    
    # Upload configurations
    config_dir = Path('configs')
    upload_directory(args.bucket, config_dir, 'configs')
    
    print("Upload complete!")

if __name__ == '__main__':
    main()
