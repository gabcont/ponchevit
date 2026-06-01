using System.Collections.Generic;

namespace Ponchevit.Data;

/// <summary>
/// Read/write repository for the per-project Revit-material → Covenin-value-ID mapping.
/// The only implementation (ExtensibleStorageMaterialMappingRepository) lives in Revit/
/// because it depends on Revit ExtensibleStorage; this interface stays in Data/ to keep
/// Domain/ pure C#.
/// </summary>
public interface IMaterialMappingRepository
{
    IReadOnlyDictionary<string, string> GetAll();
    void Set(string revitMaterialName, string coveninValueId);
    void Remove(string revitMaterialName);
    void Clear();
}
