using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AnnaAtkinsFunctions.Models;
using AnnaAtkinsFunctions.Storage;

namespace AnnaAtkinsFunctions.Services
{
    public class ImageService : IImageService
    {
        private readonly IImageMetadataDB _db;
        private readonly IImageStorage _storage;

        public ImageService(IImageMetadataDB db, IImageStorage storage)
        {
            _db = db;
            _storage = storage;
        }

        public Task<ImageAnnotation[]> GetAnnotations(string imageId)
        {
            return _db.GetAnnotations(imageId);
        }

        public Task<ImageAnnotation> UpdateAnnotation(ImageAnnotation annotation)
        {
            return _db.UpdateAnnotation(annotation);
        }

        public Task<IEnumerable<ImageAnnotation>> GetImagesWithAnnotation(string annotation)
        {
            return _db.GetImagesWithAnnotation(annotation);
        }

        public Task<IEnumerable<ImageReference>> ListImages(string prefix)
        {
            return _storage.GetBlobsByPrefix(prefix);
        }

        public Task DeleteAnnotation(string imageId, string annotation)
        {
            return _db.DeleteAnnotation(imageId, annotation);
        }
    }
}
