using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;

namespace AnnaAtkinsFunctions.Models
{
    public class ImageReference
    {
        [JsonProperty("image_id")]
        public string ImageId { get; set; }

        [JsonProperty("url")]
        public string Url { get; set; }
    }
}
