from typing import NamedTuple, List
import io
import os
from dotenv import load_dotenv
from numpy import mean
import torch
from typing import Iterable
from PIL import Image
from azure.storage.blob import BlobClient, ContainerClient
from torch.utils.data import Dataset
import hashlib
from torchvision.transforms import Compose, ToTensor, FiveCrop, ToTensor, Lambda
from requests import request
import json

def list_blobs(
    connection_string: str, container_name: str, blob_prefix: str
) -> Iterable[str]:
    container_client = ContainerClient.from_connection_string(
        conn_str=connection_string, container_name=container_name
    )
    result = container_client.list_blobs(name_starts_with=blob_prefix)
    return [r.name for r in result]


class RectangleAnnotation(NamedTuple):
    left: float
    right: float
    top: float
    bottom: float


class RawScanAnnotations(NamedTuple):
    page: RectangleAnnotation
    swatches: RectangleAnnotation


def get_annotations(annotation_api_host: str, annotation_api_key: str, storage_connection_string:str, storage_container_name:str, blob_prefix:str):
    blob_names = list_blobs(
            storage_connection_string, storage_container_name, blob_prefix
        )
    results = {}
    for b in blob_names:
        _, image_name = os.path.split(b)
        image_id = image_name.split(".")[0]
        resp = request(
            "GET",
            "{}/api/images/{}/annotations?code={}&clientId=default".format(
                annotation_api_host, image_id, annotation_api_key
            ),
        )
        if resp.status_code < 400:
            annotations = resp.json()
            if len(annotations) > 0:
                page_left = mean(list(
                    [
                        float(a["x"])
                        for a in annotations
                        if a["annotation_name"]
                        in ["Main Page Top Left", "Main Page Bottom Left"]
                    ]
                ))
                page_right = mean(list(
                    [
                        float(a["x"])
                        for a in annotations
                        if a["annotation_name"]
                        in ["Main Page Top Right", "Main Page Bottom Right"]
                    ]
                ))
                page_top = mean(list(
                    [
                        float(a["y"])
                        for a in annotations
                        if a["annotation_name"]
                        in ["Main Page Top Left", "Main Page Top Right"]
                    ]
                ))
                page_bottom = mean(list(
                    [
                        float(a["y"])
                        for a in annotations
                        if a["annotation_name"]
                        in ["Main Page Bottom Left", "Main Page Bottom Right"]
                    ]
                ))

                swatches_left = mean(list(
                    [
                        float(a["x"])
                        for a in annotations
                        if a["annotation_name"]
                        in [
                            "Upper Swatches Top Left",
                            "Upper Swatches Bottom Left",
                            "Lower Swatches Bottom Left",
                        ]
                    ]
                ))
                swatches_right = mean(list(
                    [
                        float(a["x"])
                        for a in annotations
                        if a["annotation_name"]
                        in [
                            "Upper Swatches Top Right",
                            "Lower Swatches Top Right",
                            "Lower Swatches Bottom Right",
                        ]
                    ]
                ))
                swatches_top = mean(list(
                    [
                        float(a["y"])
                        for a in annotations
                        if a["annotation_name"]
                        in ["Upper Swatches Top Left", "Upper Swatches Top Right"]
                    ]
                ))
                swatches_bottom = mean(list(
                    [
                        float(a["y"])
                        for a in annotations
                        if a["annotation_name"]
                        in ["Lower Swatches Bottom Left", "Lower Swatches Bottom Right"]
                    ])
                )
                results[b] = RawScanAnnotations(
                    RectangleAnnotation(page_left, page_right, page_top, page_bottom),
                    RectangleAnnotation(
                        swatches_left, swatches_right, swatches_top, swatches_bottom
                    ),
                )
    return results

def RectangleFromList(l:List[float])->RectangleAnnotation:
    return RectangleAnnotation(l[0], l[1], l[2], l[3])

def RawScanAnnotationFromList(l:List[List[float]]) -> RawScanAnnotations:
    return RawScanAnnotations(RectangleFromList(l[0]), RectangleFromList(l[1]))

class AnnotatedImageDataset(Dataset):
    def __init__(
        self,
        storage_connection_string,
        storage_container_name,
        annotation_api_host,
        annotation_api_key,
        blob_prefix,
        crop_size=None,
        cache_dir=None,
    ):
        self.cache_dir = cache_dir
        self.storage_connection_string = storage_connection_string
        self.storage_container_name = storage_container_name
        cached_annotations = os.path.join(self.cache_dir, "annotations.json")
        self.annotations = None
        if os.path.exists(cached_annotations):
            try:
                with open(cached_annotations, "r") as annotation_file:
                    cached_annotations = json.load(annotation_file)
                    self.annotations = {}
                    for k in cached_annotations.keys():
                        self.annotations[k] = RawScanAnnotationFromList(cached_annotations[k])
            except:
                pass
        if self.annotations is None or not isinstance(self.annotations, dict) or len(self.annotations.keys()) == 0:
            self.annotations = get_annotations(
                annotation_api_host, annotation_api_key, storage_connection_string, storage_container_name, blob_prefix
            )
            with open(cached_annotations, "w") as annotation_file:
                json.dump(self.annotations, annotation_file)
        self.blob_names = list(self.annotations.keys())
        if crop_size:
            self.blob_transform = Compose(
                [
                    FiveCrop(crop_size),
                    Lambda(
                        lambda crops: torch.stack([ToTensor()(crop) for crop in crops])
                    ),
                ]
            )
            self.annotation_transform = Compose(
                [FiveCrop(crop_size), Lambda(lambda crops: torch.stack(crops))]
            )
        else:
            self.blob_transform = ToTensor()
            self.annotation_transform = None

    def __len__(self):
        return len(self.blob_names)

    def _get_cache_path(self, blob_name: str) -> str:
        return os.path.join(
            self.cache_dir, hashlib.md5(blob_name.encode("utf-8")).hexdigest()
        )

    def _get_blob(self, blob_name: str) -> Image.Image:
        cache_path = self._get_cache_path(blob_name)
        if os.path.exists(cache_path):
            return Image.open(cache_path)
        blob_client = BlobClient.from_connection_string(
            conn_str=self.storage_connection_string,
            container_name=self.storage_container_name,
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

    def _get_annotation_image(self, blob_name: str, blob: Image.Image) -> torch.Tensor:
        width, height = blob.size
        classes = torch.zeros((height, width), dtype=torch.long)
        annotation = self.annotations[blob_name]
        classes[
            int(annotation.page.top * height) : int(annotation.page.bottom * height),
            int(annotation.page.left * width) : int(annotation.page.right * width),
        ] = 1
        classes[
            int(annotation.swatches.top * height) : int(
                annotation.swatches.bottom * height
            ),
            int(annotation.swatches.left * width) : int(
                annotation.swatches.right * width
            ),
        ] = 2
        return classes

    def __getitem__(self, idx:int):
        blob_name = self.blob_names[idx]
        img = self._get_blob(blob_name)
        annotation = self._get_annotation_image(blob_name, img)
        return self.blob_transform(
            img
        ), annotation if self.annotation_transform == None else self.annotation_transform(
            annotation
        )

    def get_annotation_count(self) -> int:
        return 3 # background, page, swatches
