using System;
using System.IO;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Azure.WebJobs;
using Microsoft.Azure.WebJobs.Extensions.Http;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using Newtonsoft.Json;
using AnnaAtkinsFunctions.Models;
using AnnaAtkinsFunctions.Services;

namespace AnnaAtkinsFunctions
{
    public class UpdateImageAnnotation
    {
        private readonly IImageService _imageAnnotationService;

        public UpdateImageAnnotation(IImageService imageAnnotationService)
        {
            _imageAnnotationService = imageAnnotationService;
        }

        [FunctionName("UpdateImageAnnotation")]
        public async Task<IActionResult> Run(
            [HttpTrigger(AuthorizationLevel.Function, "put", Route = "images/{imageId}/annotations")] HttpRequest req,
            string imageId,
            ILogger log)
        {
            string json = await req.ReadAsStringAsync();
            var annotation = JsonConvert.DeserializeObject<ImageAnnotation>(json);
            annotation.ImageId = annotation.ImageId ?? imageId;
            annotation = await _imageAnnotationService.UpdateAnnotation(annotation);
            return new OkObjectResult(annotation);
        }
    }
}
