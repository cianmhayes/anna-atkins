using AnnaAtkinsFunctions.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Azure.Cosmos;

namespace AnnaAtkinsFunctions.Storage
{
    public class ImageMetadataDB : IImageMetadataDB
    {
        private readonly CosmosClient _client;
        private readonly Container _container;

        public ImageMetadataDB(string connectionString)
        {
            _client = new CosmosClient(connectionString);
            _container = _client.GetContainer("ImageMetadata", "Annotations");
        }

        public async Task DeleteAnnotation(string imageId, string annotation)
        {
            await _container.DeleteItemAsync<ImageAnnotation>(ImageAnnotation.MakeId(imageId, annotation), new PartitionKey(annotation));
        }

        public async Task<ImageAnnotation[]> GetAnnotations(string imageId)
        {
            List<ImageAnnotation> annotations = new List<ImageAnnotation>();
            var query = new QueryDefinition("SELECT * FROM c WHERE c.image_id = @imageId").WithParameter("@imageId", imageId);
            using (FeedIterator<ImageAnnotation> feedIterator = _container.GetItemQueryIterator<ImageAnnotation>(query))
            {
                while (feedIterator.HasMoreResults)
                {
                    FeedResponse<ImageAnnotation> response = await feedIterator.ReadNextAsync();
                    annotations.AddRange(response);
                }
            }
            return annotations.ToArray();
        }

        public async Task<IEnumerable<ImageAnnotation>> GetImagesWithAnnotation(string annotation)
        {
            List<ImageAnnotation> annotations = new List<ImageAnnotation>();
            var query = new QueryDefinition("SELECT * FROM c WHERE c.annotation_name = @annotation").WithParameter("@annotation", annotation);
            using (FeedIterator<ImageAnnotation> feedIterator = _container.GetItemQueryIterator<ImageAnnotation>(query))
            {
                while (feedIterator.HasMoreResults)
                {
                    FeedResponse<ImageAnnotation> response = await feedIterator.ReadNextAsync();
                    annotations.AddRange(response);
                }
            }
            return annotations.ToArray();
        }

        public async Task<ImageAnnotation> UpdateAnnotation(ImageAnnotation annotation)
        {
            annotation.Id = ImageAnnotation.MakeId(annotation.ImageId, annotation.AnnotationName);
            var response = await _container.UpsertItemAsync(annotation);
            return response.Resource;
        }
    }
}
