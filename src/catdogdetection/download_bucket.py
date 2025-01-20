from google.cloud import storage
import os

def list_files_in_bucket(bucket_name, prefix):
    """List all files in a bucket with a given prefix."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    return [blob.name for blob in blobs]

def download_file(bucket_name, blob_name, local_path):
    """Download a single file from a bucket."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    local_file = os.path.join(local_path, os.path.basename(blob_name))
    os.makedirs(os.path.dirname(local_file), exist_ok=True)
    blob.download_to_filename(local_file)
    print(f"Downloaded {blob_name} to {local_file}")

def download_files_with_prefix(bucket_name, prefix, local_path):
    """Download all files from a bucket with a given prefix to a local directory."""
    files = list_files_in_bucket(bucket_name, prefix)
    for file in files:
        download_file(bucket_name, file, local_path)

if __name__ == "__main__":
    bucket_name = "catdog-data"
    cats_prefix = "data/raw/cats/"
    dogs_prefix = "data/raw/dogs/"
    cats_local_path = "images/cats/"
    dogs_local_path = "images/dogs/"

    download_files_with_prefix(bucket_name, cats_prefix, cats_local_path)
    download_files_with_prefix(bucket_name, dogs_prefix, dogs_local_path)
