using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AnnaAtkinsFunctions.Models;

namespace AnnaAtkinsFunctions.Storage
{
    public interface IImageMetadataDB
    {
        Task<ImageAnnotation[]> GetAnnotations(string imageId);

        Task<ImageAnnotation> UpdateAnnotation(ImageAnnotation annotation);

        Task<IEnumerable<ImageAnnotation>> GetImagesWithAnnotation(string annotation);

        Task DeleteAnnotation(string imageId, string annotation);
    }
}
