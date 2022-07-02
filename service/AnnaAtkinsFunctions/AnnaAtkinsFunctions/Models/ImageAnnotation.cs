using System;
using System.Collections.Generic;
using System.Security.Cryptography;
using System.Linq;
using System.Text;
using System.Text.Encodings;
using System.Threading.Tasks;
using Newtonsoft.Json;
using Microsoft.AspNetCore.WebUtilities;

namespace AnnaAtkinsFunctions.Models
{
    public class ImageAnnotation
    {
        public static string MakeId(string image_id, string annotation_name)
        {
            byte[] fullKey = Encoding.UTF8.GetBytes(image_id.ToLower() + "|" + annotation_name.ToLower());
            byte[] hashedKey = SHA1CryptoServiceProvider.HashData(fullKey);
            return WebEncoders.Base64UrlEncode(hashedKey);
        }

        [JsonProperty("id")]
        public string Id { get; set; }

        [JsonProperty("image_id")]
        public string ImageId { get; set; }

        [JsonProperty("annotation_name")]
        public string AnnotationName { get; set; }

        [JsonProperty("x", NullValueHandling = NullValueHandling.Ignore)]
        public float? X { get; set; }

        [JsonProperty("y", NullValueHandling = NullValueHandling.Ignore)]
        public float? Y { get; set; }

        [JsonProperty("_etag")]
        public string Etag { get; set; }
    }
}
