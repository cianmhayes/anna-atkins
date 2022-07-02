using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Azure.Storage.Blobs;
using Azure.Storage.Blobs.Models;
using AnnaAtkinsFunctions.Models;

namespace AnnaAtkinsFunctions.Storage
{
    public class ImageStorage : IImageStorage
    {
        private readonly BlobContainerClient _blobContainerClient;

        public ImageStorage(string connectionString)
        {
            _blobContainerClient = new BlobContainerClient(connectionString, "anna-atkins");
        }

        public async Task<IEnumerable<ImageReference>> GetBlobsByPrefix(string prefix)
        {
            DateTimeOffset sasExpiration = DateTimeOffset.UtcNow + TimeSpan.FromDays(30);
            List<ImageReference> blobs = new List<ImageReference>();
            var blobQuery = _blobContainerClient.GetBlobsAsync(BlobTraits.None, BlobStates.None, prefix);
            await foreach (BlobItem blobItem in blobQuery)
            {
                string[] pathParts = blobItem.Name.Split('/', '\\');
                string fileName = pathParts[pathParts.Length - 1];
                string imageId = fileName.Split('.')[0];
                BlobClient blobClient = _blobContainerClient.GetBlobClient(blobItem.Name);
                Uri blobUrl = blobClient.GenerateSasUri(Azure.Storage.Sas.BlobSasPermissions.Read, sasExpiration);
                blobs.Add(new ImageReference() { ImageId = imageId, Url = blobUrl.ToString() });
            }
            return blobs;
        }
    }
}
