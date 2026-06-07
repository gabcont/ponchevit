using System;
using System.Collections.Generic;
using System.Linq;
using Autodesk.Revit.DB;
using Ponchevit.Domain.Codificacion;
using Ponchevit.Revit.SharedParameters;

namespace Ponchevit.Revit.Codificacion;

/// <summary>
/// Walks the active document and returns one CodificacionSummary per family type that
/// has at least one placed instance. Only the 10 supported categories are included.
/// </summary>
public sealed class ProjectInventoryReader
{
    // Category descriptor: BuiltInCategory, optional area parameter (null → count), display name.
    private static readonly (BuiltInCategory Bic, BuiltInParameter? AreaParam, string DisplayName)[] SupportedCategories =
    [
        (BuiltInCategory.OST_Walls,              BuiltInParameter.HOST_AREA_COMPUTED, "Muros"),
        (BuiltInCategory.OST_Floors,             BuiltInParameter.HOST_AREA_COMPUTED, "Pisos"),
        (BuiltInCategory.OST_Ceilings,           BuiltInParameter.HOST_AREA_COMPUTED, "Cielos rasos"),
        (BuiltInCategory.OST_Roofs,              BuiltInParameter.HOST_AREA_COMPUTED, "Cubiertas"),
        (BuiltInCategory.OST_Doors,              null,                                "Puertas"),
        (BuiltInCategory.OST_Windows,            null,                                "Ventanas"),
        (BuiltInCategory.OST_Columns,            null,                                "Columnas"),
        (BuiltInCategory.OST_StructuralFraming,  null,                                "Estructura"),
        (BuiltInCategory.OST_Stairs,             null,                                "Escaleras"),
        (BuiltInCategory.OST_GenericModel,       null,                                "Modelo genérico"),
    ];

    private const double SqFtToSqM = 0.0929;

    public IReadOnlyList<CodificacionSummary> Read(Document doc)
    {
        if (doc == null) throw new ArgumentNullException(nameof(doc));

        var result = new List<CodificacionSummary>();

        foreach (var (bic, areaParam, displayName) in SupportedCategories)
        {
            var instances = new FilteredElementCollector(doc)
                .OfCategory(bic)
                .WhereElementIsNotElementType()
                .ToElements();

            if (instances.Count == 0)
                continue;

            var byTypeId = instances.GroupBy(el => el.GetTypeId().Value);

            foreach (var group in byTypeId)
            {
                long typeIdValue = group.Key;
                if (typeIdValue == ElementId.InvalidElementId.Value)
                    continue;

                var typeEl = doc.GetElement(new ElementId(typeIdValue));
                if (typeEl == null)
                    continue;

                var groupInstances = group.ToList();

                string codigo = typeEl.LookupParameter(CoveninParameters.CodigoCompletoName)?.AsString() ?? string.Empty;
                bool isCodified = !string.IsNullOrWhiteSpace(codigo) && codigo.Length == 10;
                string? codigoCompleto = isCodified ? codigo : null;

                string familyTypeName = typeEl is FamilySymbol fs
                    ? fs.Family.Name + " : " + fs.Name
                    : typeEl.Name;

                double quantity;
                string unit;
                if (areaParam.HasValue)
                {
                    double sumSqFt = groupInstances.Sum(
                        inst => inst.get_Parameter(areaParam.Value)?.AsDouble() ?? 0.0);
                    quantity = sumSqFt * SqFtToSqM;
                    unit = "m²";
                }
                else
                {
                    quantity = groupInstances.Count;
                    unit = "ud";
                }

                result.Add(new CodificacionSummary(
                    FamilyTypeName:     familyTypeName,
                    CategoryDisplayName: displayName,
                    CodigoCompleto:     codigoCompleto,
                    InstanceCount:      groupInstances.Count,
                    QuantityValue:      quantity,
                    QuantityUnit:       unit,
                    IsCodified:         isCodified,
                    SampleInstanceId:   groupInstances[0].Id.Value));
            }
        }

        return result
            .OrderByDescending(s => s.IsCodified)
            .ThenBy(s => s.FamilyTypeName)
            .ToList();
    }
}
