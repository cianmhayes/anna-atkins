using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AnnaAtkinsFunctions.Models;
using AnnaAtkinsFunctions.Services;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Azure.WebJobs;
using Microsoft.Azure.WebJobs.Extensions.Http;
using Microsoft.Extensions.Logging;

namespace AnnaAtkinsFunctions.Functions
{
    public class GetRandomUnannotatedImage
    {
        private readonly IImageService _imageAnnotationService;

        public GetRandomUnannotatedImage(IImageService imageAnnotationService)
        {
            _imageAnnotationService = imageAnnotationService;
        }

        [FunctionName("GetRandomUnannotatedImage")]
        public async Task<IActionResult> Run(
            [HttpTrigger(AuthorizationLevel.Function, "get", Route = "images/random-unannotated")] HttpRequest req,
            ILogger log)
        {
            string targetAnnotations = req.Query["target-annotations"];
            string missingAnyOrAll= req.Query["missing-any-or-all"];
            string[] annotations = targetAnnotations.Split(',', StringSplitOptions.RemoveEmptyEntries);

            IEnumerable<ImageReference> images = await _imageAnnotationService.ListImages("unprocessed_1000px/");
            IEnumerable<string> completedImages = null;
            foreach (string annotation in annotations)
            {
                IEnumerable<string> annotatedImages = (await _imageAnnotationService.GetImagesWithAnnotation(annotation)).Select(a => a.ImageId);
                if (completedImages == null)
                {
                    completedImages = new List<string>(annotatedImages);
                }
                else
                {
                    if (missingAnyOrAll == "all")
                    {
                        completedImages = completedImages.Intersect(annotatedImages);
                    }
                    else
                    {
                        completedImages = completedImages.Union(annotatedImages);
                    }
                }
            }
            ImageReference ir = images.Where(i => !completedImages.Contains(i.ImageId)).OrderBy(i => Random.Shared.Next()).FirstOrDefault();
            return new OkObjectResult(ir);
        }
    }
}
