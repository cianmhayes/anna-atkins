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
    public class ListImages
    {
        private readonly IImageService _imageAnnotationService;

        public ListImages(IImageService imageAnnotationService)
        {
            _imageAnnotationService = imageAnnotationService;
        }

        [FunctionName("ListImages")]
        public async Task<IActionResult> Run(
            [HttpTrigger(AuthorizationLevel.Function, "get", Route = "images")] HttpRequest req,
            ILogger log)
        {
            var images = await _imageAnnotationService.ListImages("unprocessed_1000px/");
            return new OkObjectResult(images);
        }
    }
}
