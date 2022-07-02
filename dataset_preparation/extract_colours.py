from importlib.metadata import metadata
import io
import os
from typing import Iterable, List, Tuple
from PIL import Image
from colormath.color_objects import LabColor, XYZColor
from colormath.color_conversions import convert_color
from scipy import ndimage
import numpy as np
from skimage import color
from skimage.transform import resize
import matplotlib.pyplot as plt
from ransac import RansacBase, RansacOptions
from geometric_median import geometric_median
from azure.storage.blob import BlobClient, ContentSettings, ContainerClient
from multiprocessing import Pool, cpu_count
from functools import partial
from datetime import datetime
import json


ALL_SWATCHES = [
    LabColor(39.12, 13.24, 15.07),     # Brown
    LabColor(65.43, 18.11, 18.72),     # Cream
    LabColor(49.87, -4.34, -22.29),    # Sky Blue
    LabColor(44.26, -13.80, 22.85),    # Forest Green
    LabColor(55.56, 9.82, -24.49),     # Lavendar
    LabColor(70.82, -33.43, -0.35),    # Mint
    LabColor(63.51, 34.26, 59.60),     # Rust
    LabColor(39.92, 11.81, -46.07),    # Indigo
    LabColor(52.24, 48.55, 18.51),     # Salmon
    LabColor(97.06, -0.40, 1.13),      # 10
    LabColor(92.02, -0.60, 0.23),      # 11 (A)
    LabColor(87.34, -0.75, 0.21),      # 12
    LabColor(82.14, -1.06, 0.43),      # 13
    LabColor(72.06, -1.19, 0.28),      # 14
    LabColor(62.15, -1.07, 0.19),      # 15
    LabColor(49.25, -0.16, 0.01),      # 16 (M)
    LabColor(38.62, -0.8, -0.04),      # 17
    LabColor(28.86, 0.54, 0.60),       # 18 (B)
    LabColor(16.19, -0.05, 0.73),      # 19
    LabColor(8.29, -0.81, 0.19),       # 20
    LabColor(3.44, -0.23, 0.49),       # 21
    LabColor(31.41, 20.98, -19.43),    # Purple
    LabColor(72.46, -24.45, 55.93),    # other mint?
    LabColor(72.95, 16.83, 68.80),     # Mustard
    LabColor(29.37, 13.06, -49.49),    # Blue
    LabColor(54.91, -38.91, 30.77),    # Green
    LabColor(43.96, 52.00, 30.01),     # Red
    LabColor(82.74, 3.45, 81.29),      # Yellow
    LabColor(52.79, 50.88, -12.72),    # Pink
    LabColor(50.87, -27.17, -29.46)    # Cyan
    ]

CYANOTYPE_SWATCHES = [ 2, 7, 29 ]

storage_connection_string = os.environ.get("STORAGE_CONNECTION_STRING")
storage_account_container = "anna-atkins"

def save_colorized_image(image, path, mono=False):
    cm = plt.get_cmap('Greys' if mono else 'viridis')
    colored_image = cm(image)
    Image.fromarray((colored_image[:, :, :3] * 255).astype(np.uint8)).save(path)

class ColorSwatchRansac(RansacBase):
    def __init__(self, starting_points:np.ndarray, ending_points:np.ndarray, swatch_diffs:List[np.ndarray], width:int, height:int) -> None:
        super().__init__()
        self.starting_points, self.starting_points_idx, self.starting_points_idx_p = self._prepare_endpoints_by_threshold(starting_points)
        self.ending_points, self.ending_points_idx, self.ending_points_idx_p = self._prepare_endpoints_by_threshold(ending_points)
        self.swatch_diffs = swatch_diffs
        self.width = width
        self.height = height
    
    def _prepare_endpoints_by_percentile(self, endpoints):
        endpoints /= endpoints.sum()
        endpoints = 1 - endpoints
        endpoints /= endpoints.sum()

        endpoint_indices = np.argsort(endpoints)
        endpoint_indices = endpoint_indices[::-1]
        endpoint_indices = endpoint_indices[:int(len(endpoints)*0.1)]
        endpoint_indices_p = endpoints[endpoint_indices]
        endpoint_indices_p /= endpoint_indices_p.sum()

        return endpoints, endpoint_indices, endpoint_indices_p

    def _prepare_endpoints_by_threshold(self, endpoints):
        endpoints -= endpoints.min()
        endpoints /= endpoints.max()
        endpoint_indices = np.where(endpoints < 0.1)[0]
        endpoint_indices_p = endpoints[endpoint_indices]

        endpoint_indices_p /= endpoint_indices_p.sum()
        endpoint_indices_p = 1 - endpoint_indices_p
        endpoint_indices_p /= endpoint_indices_p.sum()

        return endpoints, endpoint_indices, endpoint_indices_p

    def generate_sample_indices(self, sample_count:int) -> Tuple[np.ndarray, np.ndarray]:
        start_samples = np.random.choice(
            self.starting_points_idx,
            sample_count,
            replace=False,
            p=self.starting_points_idx_p)
        start_samples = start_samples.reshape(-1, 1)
        starting_x = start_samples % self.width
        starting_y = start_samples // self.width

        end_samples = np.random.choice(
            self.ending_points_idx,
            sample_count,
            replace=False,
            p=self.ending_points_idx_p)
        end_samples = end_samples.reshape(-1, 1)
        ending_x = end_samples % self.width
        ending_y = end_samples // self.width
        return np.concatenate([starting_y, starting_x, ending_y, ending_x], 1), np.array([])

    def test_points(self, test_points:np.ndarray, maybe_model:np.ndarray) -> np.ndarray:
        total_error = 0.0
        for i in range(len(self.swatch_diffs)):
            total_error += self.swatch_diffs[i][maybe_model[i][0]][maybe_model[i][1]]
        total_error /= len(self.swatch_diffs)
        return np.array([total_error])

    def fit_model(self, candidates:np.ndarray) -> np.ndarray:
        start_medoid = geometric_median(candidates[:, 0:2], eps=0.1)
        end_medoid = geometric_median(candidates[:, 2:], eps=0.1)

        x_delta = (end_medoid[1] - start_medoid[1]) / (len(self.swatch_diffs) - 1)
        y_delta = (end_medoid[0] - start_medoid[0]) / (len(self.swatch_diffs) - 1)
        points = []
        for i in range(len(self.swatch_diffs)):
            x = int(start_medoid[1] + (i * x_delta))
            y = int(start_medoid[0] + (i * y_delta))
            points.append(np.array([y, x]))
        return np.vstack(points)

class SegmentAligner(object):
    def __init__(self, img:np.ndarray) -> None:
        super().__init__()
        self.seg_edge = 10
        #img = io.imread(image_path)
        self.original_width = img.shape[1]
        self.original_height = img.shape[0]
        self.resize_ratio = (self.seg_edge * 200) / self.original_height
        self.width = int(img.shape[1] * self.resize_ratio)
        self.height = int(img.shape[0] * self.resize_ratio)
        resized_img = resize(img, (self.height, self.width), anti_aliasing=False)
        self.lab_image = color.rgb2lab(resized_img)

    def find_cyanotype(self, kernel_size=9, threshold=0.2):
        accumulated = None
        left = top = right = bottom = 0
        for i in CYANOTYPE_SWATCHES:
            swatch_diff = self.diff_with_swatch(i, kernel_size)
            if accumulated is None:
                accumulated = swatch_diff
            else:
                accumulated = np.minimum(accumulated, swatch_diff)
        lower_quartile = threshold
        accumulated[accumulated <= lower_quartile] = 0
        accumulated[accumulated >= lower_quartile] = 1
        accumulated -= 1
        accumulated *= -1
        x_sums = np.sum(accumulated, 0)
        x_indices = np.argwhere(x_sums > (self.height / 3))
        if len(x_indices) == 0:
            return (left, top, right, bottom)
        left = int(x_indices[0] / self.resize_ratio)
        right = int(x_indices[-1] / self.resize_ratio)
        y_sums = np.sum(accumulated, 1)
        y_indices = np.argwhere(y_sums > (self.width / 3))
        if len(y_indices) == 0:
            return (left, top, right, bottom)
        top = int(y_indices[0] / self.resize_ratio)
        bottom = int(y_indices[-1] / self.resize_ratio)
        return (left, top, right, bottom)


    def diff_with_swatch(self, color_index:int, kernel_size=None):
        delta = self.lab_image - ALL_SWATCHES[color_index].get_value_tuple()
        diff = np.sqrt((delta**2).sum(-1))
        smoothed = ndimage.maximum_filter(diff, kernel_size or self.seg_edge)
        smoothed -= smoothed.min()
        smoothed /= smoothed.max()
        return smoothed

    def find_reference_points(self):
        x_hist = None
        diffs = []
        for i in range(len(ALL_SWATCHES)):
            diff = self.diff_with_swatch(i)
            clipped_diff = np.copy(diff)
            clipped_diff[clipped_diff >= 0.2] = 1
            diffs.append(diff)
            if x_hist is None:
                x_hist = np.min(clipped_diff, axis=0)
            else:
                x_hist += np.min(clipped_diff, axis=0)
        moving_average = ndimage.uniform_filter1d(x_hist, self.seg_edge, axis=0)
        x_center = np.argsort(moving_average, axis=None)[0]
        left = int(x_center - (2 * self.seg_edge))
        right = int(x_center + (2 * self.seg_edge))
        for i in range(len(ALL_SWATCHES)):
            diffs[i] = diffs[i][:, left:right]
            diffs[i] -= diffs[i].min()
            diffs[i] /= diffs[i].max()

        diff_width = len(diffs[0][0])
        diff_height = len(diffs[0])

        start_points = np.reshape(diffs[0], -1)
        end_points = np.reshape(diffs[14], -1)
        first_rsac = ColorSwatchRansac(start_points, end_points, diffs[0:15], diff_width, diff_height)
        first_rsac_model = first_rsac.run(RansacOptions(1000, 100000, 0.05, 0, 0.2, 20))
        for i in range(len(first_rsac_model)):
            first_rsac_model[i][1] += left
        first_rsac_model = (first_rsac_model / self.resize_ratio).astype(int)

        start_points = np.reshape(diffs[15], -1)
        end_points = np.reshape(diffs[29], -1)
        second_rsac = ColorSwatchRansac(start_points, end_points, diffs[15:], diff_width, diff_height)
        second_rsac_model = second_rsac.run(RansacOptions(1000, 100000, 0.05, 0, 0.2, 20))
        for i in range(len(second_rsac_model)):
            second_rsac_model[i][1] += left
        second_rsac_model = (second_rsac_model / self.resize_ratio).astype(int)

        return np.vstack([first_rsac_model, second_rsac_model])


class ColourCorrector(object):
    def __init__(self, img:np.ndarray, use_xyz=False, use_least_squares=False) -> None:
        super().__init__()
        self.use_xyz = use_xyz
        self.use_least_squares = use_least_squares
        #img = io.imread(image_path)
        if self.use_xyz:
            self.source_img = color.rgb2xyz(img)
        else:
            self.source_img = color.rgb2lab(img)
        self.patch_range = 10
        self.reference_matrix = self._get_reference_matrix()

    def _get_reference_matrix(self):
        swatches = []
        for ls in ALL_SWATCHES:
            if self.use_xyz:
                c = convert_color(ls, XYZColor)
                swatches.append(c.get_value_tuple())
            else:
                swatches.append(ls.get_value_tuple())
        return np.array(swatches)

    def _get_reference_value(self, y, x):
        region = self.source_img[int(y-self.patch_range):int(y+self.patch_range), int(x-self.patch_range):int(x+self.patch_range)]
        return region.mean(axis=(0,1))

    def calculate_ccm(self, center_points):
        swatch_values = []
        for i in range(len(center_points)):
            swatch_values.append(self._get_reference_value(center_points[i][0], center_points[i][1]))
        swatch_values_matrix = np.vstack(swatch_values)

        swatch_values_hm = swatch_values_matrix
        if self.use_least_squares:
            return np.linalg.lstsq(swatch_values_hm, self.reference_matrix)[0]
        else:
            return np.linalg.pinv(swatch_values_hm).dot(self.reference_matrix)

    def convert_image(self, ccm):
        corrected = np.matmul(self.source_img, ccm)
        if self.use_xyz:
            return color.xyz2rgb(corrected)
        else:
            return color.lab2rgb(corrected)


def list_source_blobs(prefix:str) -> Iterable[str]:
    container_client = ContainerClient.from_connection_string(conn_str=storage_connection_string, container_name=storage_account_container)
    result = container_client.list_blobs(name_starts_with=prefix)
    return [r.name for r in result]

def get_image_from_blob(blob_path:str) -> Image.Image:
    blob_client = BlobClient.from_connection_string(conn_str=storage_connection_string, container_name=storage_account_container, blob_name=blob_path)
    blob_data = blob_client.download_blob()
    bio = io.BytesIO()
    blob_data.readinto(bio)
    return Image.open(bio)

def write_image_to_blob(img:Image.Image, blob_path:str) -> None:
    bio = io.BytesIO()
    img.save(bio, "PNG")
    bio.seek(0)
    blob_client = BlobClient.from_connection_string(conn_str=storage_connection_string, container_name=storage_account_container, blob_name=blob_path+".png")
    cs = ContentSettings(content_type="image/png")
    blob_client.upload_blob(bio, content_settings=cs)

def write_metadata(obj, blob_path:str) -> None:
    c = json.dumps(obj, indent=4)
    blob_client = BlobClient.from_connection_string(conn_str=storage_connection_string, container_name=storage_account_container, blob_name=blob_path)
    blob_client.upload_blob(c)

def blob_exists(path:str) -> bool :
    blob_client = BlobClient.from_connection_string(conn_str=storage_connection_string, container_name=storage_account_container, blob_name=path)
    return blob_client.exists()

def process_raw_blob(source_blob_path:str, output_blob_root:str) -> None:
    try:
        _process_raw_blob(source_blob_path, output_blob_root)
    except Exception as ex:
        print(ex)


def _process_raw_blob(source_blob_path:str, output_blob_root:str) -> None:
    path_parts = source_blob_path.split("/")
    file_name_parts = path_parts[-1].split(".")
    file_name = file_name_parts[0]
    metadata_blob_path = output_blob_root + "/metadata/" + file_name + ".json"
    if (blob_exists(metadata_blob_path)) :
        return
    source_img = get_image_from_blob(source_blob_path)
    img = np.array(source_img)
    sa = SegmentAligner(img)
    left, top, right, bottom = sa.find_cyanotype()
    centers = sa.find_reference_points()
    cc = ColourCorrector(img, False, False)
    ccm = cc.calculate_ccm(centers)
    converted_array = cc.convert_image(ccm)
    converted_img = Image.fromarray((converted_array * 255).astype(np.uint8))
    metadata = {
        "id": file_name,
        "crop": {
            "top": top,
            "bottom": bottom,
            "left": left,
            "right": right
        },
        "centers": [[ int(c[0]), int(c[1])] for c in centers],
        "ccm": [[ float(c_) for c_ in c ] for c in ccm],
    }
    write_metadata(metadata, metadata_blob_path)

    # save uncropped thumbnail
    resize_ratio = 1200 / converted_img.size[0]
    uncropped_thumbnail = converted_img.resize((int(converted_img.size[0] * resize_ratio),int(converted_img.size[1] * resize_ratio)))
    write_image_to_blob(uncropped_thumbnail, output_blob_root + "/uncropped_thumbnail/" + file_name)

    if left >= right or top >= bottom:
        return
    # save full res crop
    cropped_fullres = converted_img.crop((left, top, right, bottom))
    write_image_to_blob(cropped_fullres, output_blob_root + "/cropped_fullres/" + file_name)

    # save crop thumbnail
    resize_ratio = 1200 / cropped_fullres.size[0]
    cropped_thumbnail = cropped_fullres.resize((int(cropped_fullres.size[0] * resize_ratio),int(cropped_fullres.size[1] * resize_ratio)))
    write_image_to_blob(cropped_thumbnail, output_blob_root + "/cropped_thumbnail/" + file_name)

if __name__ == "__main__":
    blobs = list_source_blobs("highres_images")
    output_root = "processed_images"# + datetime.now().strftime("%Y_%m_%d_%H_%M")
    with Pool(cpu_count()) as p:
        p.map(partial(process_raw_blob, output_blob_root=output_root), blobs)
