using System.Collections.Generic;
using System.Linq;
using Ponchevit.Data;
using Ponchevit.Domain.Model;

namespace Ponchevit.Domain.Materials;

public interface IMaterialMappingResolver
{
    /// <summary>
    /// Returns the Covenin material value ID for the given Revit material name,
    /// or null when no mapping exists.
    /// </summary>
    string? Resolve(string revitMaterialName);
}

public sealed class MaterialMappingResolver : IMaterialMappingResolver
{
    private readonly IMaterialMappingRepository _repo;

    public MaterialMappingResolver(IMaterialMappingRepository repo) => _repo = repo;

    public string? Resolve(string revitMaterialName)
    {
        if (string.IsNullOrWhiteSpace(revitMaterialName))
            return null;
        var all = _repo.GetAll();
        return all.TryGetValue(revitMaterialName, out var id) ? id : null;
    }
}
