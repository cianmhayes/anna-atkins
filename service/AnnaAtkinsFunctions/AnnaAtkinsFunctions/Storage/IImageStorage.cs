using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AnnaAtkinsFunctions.Models;

namespace AnnaAtkinsFunctions.Storage
{
    public interface IImageStorage
    {
        Task<IEnumerable<ImageReference>> GetBlobsByPrefix(string prefix);
    }
}
