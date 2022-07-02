using System;
using System.IO;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Azure.WebJobs;
using Microsoft.Azure.WebJobs.Extensions.Http;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using Newtonsoft.Json;
using AnnaAtkinsFunctions.Services;

namespace AnnaAtkinsFunctions.Functions
{
    public class GetImageAnnotations
    {

        private readonly IImageService _imageAnnotationService;

        public GetImageAnnotations(IImageService imageAnnotationService)
        {
            _imageAnnotationService = imageAnnotationService;
        }

        [FunctionName("GetImageAnnotations")]
        public async Task<IActionResult> Run(
            [HttpTrigger(AuthorizationLevel.Function, "get", Route = "images/{imageId}/annotations")] HttpRequest req,
            string imageId,
            ILogger log)
        {
            var annotations = await _imageAnnotationService.GetAnnotations(imageId);
            return new OkObjectResult(annotations);
        }
    }
}
