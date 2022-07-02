import io
from typing import Iterable
from PIL import Image
from azure.storage.blob import BlobClient, ContentSettings, ContainerClient

class ImageBlobStorage(object):
    def __init__(self, connection_string, container) -> None:
        self.connection_string = connection_string
        self.container = container

    def list_source_blobs(self, prefix:str) -> Iterable[str]:
        container_client = ContainerClient.from_connection_string(conn_str=self.connection_string, container_name=self.container)
        result = container_client.list_blobs(name_starts_with=prefix)
        return [r.name for r in result]

    def get_image_from_blob(self, blob_path:str) -> Image.Image:
        blob_client = BlobClient.from_connection_string(conn_str=self.connection_string, container_name=self.container, blob_name=blob_path)
        blob_data = blob_client.download_blob()
        bio = io.BytesIO()
        blob_data.readinto(bio)
        return Image.open(bio)

    def delete_blob(self, blob_path:str) -> None:
        blob_client = BlobClient.from_connection_string(conn_str=self.connection_string, container_name=self.container, blob_name=blob_path)
        blob_client.delete_blob()

    def write_image_to_blob(self, img:Image.Image, blob_path:str) -> None:
        bio = io.BytesIO()
        img.save(bio, "PNG")
        bio.seek(0)
        blob_client = BlobClient.from_connection_string(conn_str=self.connection_string, container_name=self.container, blob_name=blob_path)
        cs = ContentSettings(content_type="image/png")
        blob_client.upload_blob(bio, content_settings=cs)