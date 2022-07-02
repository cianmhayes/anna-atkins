using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Azure.Functions.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection;
using AnnaAtkinsFunctions.Services;
using AnnaAtkinsFunctions.Storage;

[assembly: FunctionsStartup(typeof(AnnaAtkinsFunctions.Startup))]

namespace AnnaAtkinsFunctions
{
    public class Startup : FunctionsStartup
    {
        public override void Configure(IFunctionsHostBuilder builder)
        {
            builder.Services.AddSingleton<IImageService, ImageService>();
            builder.Services.AddSingleton(s =>
            {
                string connectionString = builder.GetContext().Configuration["ConnectionStrings:IMAGE_METADATA_CONNECTION_STRING"];
                if (string.IsNullOrWhiteSpace(connectionString))
                {
                    throw new Exception("Missing value for IMAGE_METADATA_CONNECTION_STRING");
                }
                return (IImageMetadataDB)new ImageMetadataDB(connectionString);
            });
            builder.Services.AddSingleton(s =>
            {
                string connectionString = builder.GetContext().Configuration["ConnectionStrings:IMAGE_STORAGE_CONNECTION_STRING"];
                if (string.IsNullOrWhiteSpace(connectionString))
                {
                    throw new Exception("Missing value for IMAGE_STORAGE_CONNECTION_STRING");
                }
                return (IImageStorage)new ImageStorage(connectionString);
            });
        }
    }
}
