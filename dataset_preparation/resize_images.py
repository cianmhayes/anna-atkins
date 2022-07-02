import os
from tkinter import Image
from image_blobs import ImageBlobStorage
from PIL import Image
from multiprocessing import Pool, cpu_count

storage_connection_string = os.environ.get("STORAGE_CONNECTION_STRING")
storage_account_container = "anna-atkins"

def resize_image(b):
    target_short_side = 1000
    storage = ImageBlobStorage(storage_connection_string, storage_account_container)
    img = storage.get_image_from_blob(b)
    rescale = target_short_side / float(img.size[0])
    resized = img.resize((int(img.size[0] * rescale), int(img.size[1] * rescale)), Image.BICUBIC)
    filename = b.split("/")[-1]
    name_part = filename.split(".")[0]
    target_blob = "unprocessed_1000px/{}.png".format(name_part)
    storage.write_image_to_blob(resized, target_blob)
        
if __name__ == "__main__":
    storage = ImageBlobStorage(storage_connection_string, storage_account_container)
    blobs = storage.list_source_blobs("highres_images/")
    with Pool(cpu_count() - 2) as p:
        p.map(resize_image, blobs)

    
        
        