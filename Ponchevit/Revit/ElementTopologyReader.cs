using System.Collections.Generic;
using Autodesk.Revit.DB;
using Ponchevit.Domain.Materials;
using Ponchevit.Domain.Matching;

namespace Ponchevit.Revit;

/// <summary>
/// Extracts a Revit element's structural topology (category, compound layers, key dimensions)
/// into a pure-C# ElementTopology snapshot. Material names are routed through
/// IMaterialMappingResolver so callers receive Covenin value IDs alongside the raw names.
/// All dimension values are in Revit internal units (decimal feet).
/// </summary>
public sealed class ElementTopologyReader
{
    private readonly IMaterialMappingResolver _materialResolver;

    public ElementTopologyReader(IMaterialMappingResolver materialResolver)
        => _materialResolver = materialResolver;

    public ElementTopology Read(Element element)
    {
        string category = element.Category?.Name ?? string.Empty;
        var layers = ReadLayers(element);
        var dimensions = ReadDimensions(element);
        return new ElementTopology(category, layers, dimensions);
    }

    private IReadOnlyList<MaterialLayer> ReadLayers(Element element)
    {
        var result = new List<MaterialLayer>();

        if (element is not Wall wall)
            return result;

        CompoundStructure? cs = wall.WallType?.GetCompoundStructure();
        if (cs == null)
            return result;

        foreach (CompoundStructureLayer layer in cs.GetLayers())
        {
            string matName = string.Empty;
            if (layer.MaterialId != ElementId.InvalidElementId)
                matName = element.Document.GetElement(layer.MaterialId)?.Name ?? string.Empty;

            string? coveninId = string.IsNullOrEmpty(matName)
                ? null
                : _materialResolver.Resolve(matName);

            result.Add(new MaterialLayer(matName, coveninId, layer.Width));
        }

        return result;
    }

    private static IReadOnlyDictionary<string, double> ReadDimensions(Element element)
    {
        var dims = new Dictionary<string, double>();

        if (element is Wall wall)
        {
            Parameter? height = wall.get_Parameter(BuiltInParameter.WALL_USER_HEIGHT_PARAM);
            if (height != null)
                dims["Altura"] = height.AsDouble();

            if (wall.WallType != null)
                dims["Espesor"] = wall.WallType.Width;
        }

        return dims;
    }
}
