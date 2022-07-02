import json
import os
import requests
from azure.storage.blob import ContainerClient, ContentSettings
from collections import namedtuple
from hashlib import sha256
from multiprocessing import Pool, cpu_count
from selenium.webdriver import Edge


chunk_size = 4 * 1024 * 1024

storage_connection_string = os.environ("STORAGE_CONNECTION_STRING")

list_pages = [
    "https://digitalcollections.nypl.org/collections/photographs-of-british-algae-cyanotype-impressions?format=html&id=photographs-of-british-algae-cyanotype-impressions&per_page=250&page=1#/?tab=navigation&scroll=250",
    "https://digitalcollections.nypl.org/collections/photographs-of-british-algae-cyanotype-impressions?format=html&id=photographs-of-british-algae-cyanotype-impressions&per_page=250&page=2#/?tab=navigation&scroll=250"
]

def get_service_connection() -> ContainerClient:
    return ContainerClient.from_connection_string(storage_connection_string,"anna-atkins")

def get_item_pages_from_list_pages(list_pages):
    item_pages = []
    driver = Edge()
    for p in list_pages:
        driver.get(p)
        items = driver.find_elements_by_xpath("//div[@class = 'item']/a")
        for i in items:
            current_item = {
                "id": id,
                "title": i.get_attribute('title'),
                "url": i.get_attribute('href')
            }
            current_item["id"] = sha256(current_item["url"]).hexdigest()
            print("{title} : {url}".format_map(current_item))
            item_pages.append(current_item)
    return item_pages

def get_download_links_for_item_pages(item_pages):
    results = []
    driver = Edge()
    for item_page in item_pages:
        driver.get(item_page["url"])
        original_link = driver.find_element_by_xpath("//a[@class = 'deriv-link original-link']").get_attribute('href')
        highres_link = driver.find_element_by_xpath("//a[@class = 'deriv-link highres']").get_attribute('href')
        item_page["image_urls"] = [
            {
                "resolution" : "original",
                "url" : original_link
            },
            {
                "resolution" : "highres",
                "url" : highres_link
            }
        ]
        results.append(item_page)
    return results

DownloadJob = namedtuple("DownloadJob", "source_url local_path blob_path")

def make_download_jobs(items):
    service = get_service_connection()
    jobs = []
    for item in items:
        for i in range(len(item["image_urls"])):
            ext = None
            if item["image_urls"][i]["resolution"] == "original":
                ext = ".jpeg"
            elif item["image_urls"][i]["resolution"] == "highres":
                ext = ".tiff"
            else:
                raise Exception("bad resolution")
            id = item["id"]
            local_path = id + ext
            blob_path = item["image_urls"][i]["resolution"] + "_images/" + id + ext
            if service.get_blob_client(blob_path).exists():
                continue
            jobs.append(DownloadJob(item["image_urls"][i]["url"],local_path,blob_path))
    return jobs


def download_item(item:DownloadJob):
    service = get_service_connection()
    r = requests.get(item.source_url, stream=True)
    if r.headers['content-type'] != "image/tiff" and r.headers['content-type'] != "image/jpeg":
        return
    with open(item.local_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            f.write(chunk)
    with open(item.local_path, 'rb') as data:
        service.upload_blob(name=item.blob_path, data=data, content_settings=ContentSettings(content_type=r.headers['content-type']))
    print(item.local_path)
    os.remove(item.local_path)


if __name__ == "__main__":
    item_pages = get_item_pages_from_list_pages(list_pages)
    download_links = get_download_links_for_item_pages(item_pages)
    index = None
    with open("index.json", "r") as index_file:
        index = json.load(index_file)
    jobs = make_download_jobs(index)
    print("{} items to download".format(len(jobs)))
    with Pool(cpu_count()) as p:
        p.map(download_item, jobs)
