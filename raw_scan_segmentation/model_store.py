from typing import Dict
from azure.storage.blob import BlobClient, ContainerClient
import json
import os
import hashlib

class ModelStore(object):
    def __init__(
        self, connection_string: str, container_name: str, cache_dir: str
    ) -> None:
        self.connection_string = connection_string
        self.container_name = container_name
        self.cache_dir = cache_dir

    def _get_cache_path(self, blob_name: str, etag:str) -> str:
        return os.path.join(
            self.cache_dir, hashlib.md5((blob_name + etag).encode("utf-8")).hexdigest()
        )

    def _download_blob_to_cache(self, blob_name: str) -> str:
        blob_client = BlobClient.from_connection_string(
            conn_str=self.connection_string,
            container_name=self.container_name,
            blob_name=blob_name,
        )
        cached_path = self._get_cache_path(blob_name, blob_client.get_blob_properties().etag)
        if os.path.exists(cached_path):
            return cached_path
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        blob_data = blob_client.download_blob()
        with open(cached_path, "wb") as cached_file:
            cached_file.write(blob_data.readall())
        return cached_path

    def publish(
        self,
        path_to_checkpoint: str,
        path_to_onnx: str,
        model_type: str,
        model_subtype: str,
        model_metadata: Dict[str, str],
    ) -> None:
        container_client = ContainerClient.from_connection_string(
            conn_str=self.connection_string, container_name=self.container_name
        )
        with open(path_to_checkpoint, "rb") as checkpoint_file:
            container_client.upload_blob(
                model_type + "/" + model_subtype + "/checkpoint.pt",
                checkpoint_file.read(),
                overwrite=True,
            )
        with open(path_to_onnx, "rb") as checkpoint_file:
            container_client.upload_blob(
                model_type + "/" + model_subtype + "/model.onnx",
                checkpoint_file.read(),
                overwrite=True,
            )
        container_client.upload_blob(
            model_type + "/" + model_subtype + "/metadaa.json",
            json.dumps(model_metadata, indent=4),
            overwrite=True,
        )

    def get_checkpoint(self, model_type: str, model_subtype: str) -> str:
        blob_name = model_type + "/" + model_subtype + "/checkpoint.pt"
        return self._download_blob_to_cache(blob_name)
