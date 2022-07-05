import io
import os
import torch
from typing import Iterable
from PIL import Image
from azure.storage.blob import BlobClient, ContainerClient
from torch.utils.data import Dataset
import hashlib
from torchvision.transforms import Compose, ToTensor, FiveCrop, ToTensor, Lambda


def list_blobs(
    connection_string: str, container_name: str, blob_prefix: str
) -> Iterable[str]:
    container_client = ContainerClient.from_connection_string(
        conn_str=connection_string, container_name=container_name
    )
    result = container_client.list_blobs(name_starts_with=blob_prefix)
    return [r.name for r in result]


class BlobStorageImageDataset(Dataset):
    def __init__(
        self,
        connection_string,
        container_name,
        blob_prefix,
        crop_size=None,
        cache_dir=None,
    ):
        self.connection_string = connection_string
        self.container_name = container_name
        self.blob_names = list_blobs(connection_string, container_name, blob_prefix)
        self.cache_dir = cache_dir
        if crop_size:
            self.transform = Compose(
                [
                    FiveCrop(crop_size),
                    Lambda(
                        lambda crops: torch.stack(
                            [ToTensor()(crop) for crop in crops]
                        )
                    ),
                ]
            )
        else:
            self.transform = ToTensor()

    def __len__(self):
        return len(self.blob_names)

    def _get_cache_path(self, blob_name: str) -> str:
        return os.path.join(
            self.cache_dir, hashlib.md5(blob_name.encode("utf-8")).hexdigest()
        )

    def _get_blob(self, blob_name: str) -> Image.Image:
        blob_client = BlobClient.from_connection_string(
            conn_str=self.connection_string,
            container_name=self.container_name,
            blob_name=blob_name,
        )
        blob_data = blob_client.download_blob()
        bio = io.BytesIO()
        blob_data.readinto(bio)
        img = Image.open(bio)
        if self.cache_dir:
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)
            img.save(self._get_cache_path(blob_name), "PNG")
        return img

    def __getitem__(self, idx):
        blob_name = self.blob_names[idx]
        cache_path = self._get_cache_path(blob_name)
        img = None
        if os.path.exists(cache_path):
            img = Image.open(cache_path)
        else:
            img = self._get_blob(blob_name)
        return self.transform(img)
