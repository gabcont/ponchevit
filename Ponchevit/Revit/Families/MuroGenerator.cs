using System;
using System.Collections.Generic;
using System.Linq;
using Autodesk.Revit.DB;
using Ponchevit.Data;
using Ponchevit.Revit.SharedParameters;

namespace Ponchevit.Revit.Families;

/// <summary>
/// Generates a code-compliant WallType in the active Revit document.
/// Duplicates a basic WallType, assigns the correct compound structure (material +
/// thickness from the user-selected COVENIN values), and writes the 4 shared parameters.
/// Must be called inside a caller-supplied Transaction (see
/// <see cref="FamilyGenerationOrchestrator"/> which opens that transaction).
/// </summary>
public sealed class MuroGenerator : IFamilyGenerator
{
    private readonly IMaterialMappingRepository _materialMappingRepo;

    public MuroGenerator(IMaterialMappingRepository materialMappingRepo)
    {
        _materialMappingRepo = materialMappingRepo
            ?? throw new ArgumentNullException(nameof(materialMappingRepo));
    }

    public BuiltInCategory SupportedCategory => BuiltInCategory.OST_Walls;

    /// <inheritdoc/>
    public bool IsGenerable(string? codigoPrefix)
        => codigoPrefix?.StartsWith("E41", StringComparison.OrdinalIgnoreCase) == true;

    /// <summary>
    /// Creates a WallType with the correct CompoundStructure, then writes the 4 COVENIN
    /// shared parameters on the type element.
    /// Requires a transaction to already be open on the Document.
    ///
    /// Throws <see cref="InvalidOperationException"/> when:
    ///   - No Basic WallType exists in the document.
    ///   - A selected COVENIN material has no matching entry in the material mapping.
    ///   - The mapped Revit material name cannot be found in the document.
    /// </summary>
    public void Generate(Document doc, GeneratorInput input)
    {
        if (doc   == null) throw new ArgumentNullException(nameof(doc));
        if (input == null) throw new ArgumentNullException(nameof(input));

        // ── 1. Resolve Revit material ──────────────────────────────────────────
        var allMappings = _materialMappingRepo.GetAll();

        ElementId revitMaterialElementId = ElementId.InvalidElementId;
        foreach (var kvp in input.SelectedValores)
        {
            var valor = kvp.Value;
            var mappingEntry = allMappings
                .FirstOrDefault(m => string.Equals(
                    m.Value, valor.IdValor, StringComparison.OrdinalIgnoreCase));

            if (mappingEntry.Key == null)
                continue; // Non-material column (e.g. thickness) — skip, don't throw.

            revitMaterialElementId = FindRevitMaterialId(doc, mappingEntry.Key);
            break;
        }

        if (revitMaterialElementId == ElementId.InvalidElementId)
            throw new InvalidOperationException(
                "Mapear este material en 'Mapeo de Materiales' antes de crear la familia.");

        // ── 2. Resolve thickness ───────────────────────────────────────────────
        double thicknessInFeet = 0.0;
        if (input.NumericValues.Count > 0)
        {
            // User specified an exact value for a range option.
            double metres = input.NumericValues.Values.First();
            thicknessInFeet = ToFeet(metres);
        }
        else
        {
            // Find the first SelectedValor with a numeric value that is NOT the material
            // (exact-value columns like espesor have NumMin == NumMax).
            foreach (var kvp in input.SelectedValores)
            {
                var valor = kvp.Value;
                if (valor.NumMin == null) continue;

                // Skip material columns (they are in the material mapping).
                bool isMaterial = allMappings.Values.Any(v =>
                    string.Equals(v, valor.IdValor, StringComparison.OrdinalIgnoreCase));
                if (isMaterial) continue;

                double rawValue = valor.NumMin.Value;
                double metres   = ConvertToMetres(rawValue, valor.Unidad);
                thicknessInFeet = ToFeet(metres);
                break;
            }
        }

        // Enforce a minimum so Revit does not reject the CompoundStructure.
        if (thicknessInFeet <= 0.0)
            thicknessInFeet = ToFeet(0.1); // 10 cm default

        // ── 3. Duplicate WallType ──────────────────────────────────────────────
        string newTypeName = $"COVENIN {input.Codigo.Value} — {input.Descripcion}";

        WallType newWallType = ResolveWallType(doc, newTypeName);

        // ── 4. Build CompoundStructure ────────────────────────────────────────
        if (revitMaterialElementId != ElementId.InvalidElementId)
        {
            var layer = new CompoundStructureLayer(
                thicknessInFeet,
                MaterialFunctionAssignment.Structure,
                revitMaterialElementId);

            CompoundStructure cs = CompoundStructure.CreateSimpleCompoundStructure(
                new List<CompoundStructureLayer> { layer });

            newWallType.SetCompoundStructure(cs);
        }

        // ── 5. Write shared params on WallType ────────────────────────────────
        SharedParameterWriter.Write(
            newWallType,
            input.Codigo,
            input.Capitulo,
            input.Subcapitulo ?? string.Empty,
            input.Seccion     ?? string.Empty);
    }

    // ── Private helpers ────────────────────────────────────────────────────────

    /// <summary>
    /// Returns an existing WallType with the target name, or duplicates the first
    /// Basic WallType found and renames it.
    /// </summary>
    private static WallType ResolveWallType(Document doc, string targetName)
    {
        // Check for an existing WallType with the same name.
        var existing = new FilteredElementCollector(doc)
            .OfClass(typeof(WallType))
            .Cast<WallType>()
            .FirstOrDefault(wt => string.Equals(wt.Name, targetName, StringComparison.OrdinalIgnoreCase));

        if (existing != null)
            return existing;

        // Duplicate the first Basic WallType.
        var baseType = new FilteredElementCollector(doc)
            .OfClass(typeof(WallType))
            .Cast<WallType>()
            .FirstOrDefault(wt => wt.Kind == WallKind.Basic);

        if (baseType == null)
            throw new InvalidOperationException(
                "No Basic WallType found in the document. " +
                "Load at least one basic wall family before running Agregar Familia.");

        return (WallType)baseType.Duplicate(targetName);
    }

    /// <summary>
    /// Finds a Revit Material element by name using FilteredElementCollector.
    /// Throws InvalidOperationException if not found.
    /// </summary>
    private static ElementId FindRevitMaterialId(Document doc, string materialName)
    {
        var material = new FilteredElementCollector(doc)
            .OfClass(typeof(Material))
            .Cast<Material>()
            .FirstOrDefault(m => string.Equals(m.Name, materialName, StringComparison.OrdinalIgnoreCase));

        if (material == null)
            throw new InvalidOperationException(
                $"Material de Revit '{materialName}' no encontrado en el proyecto. " +
                "Verifique que el material existe y que el mapeo en 'Mapeo de Materiales' es correcto.");

        return material.Id;
    }

    /// <summary>
    /// Converts metres to Revit internal feet using the official Revit API.
    /// </summary>
    private static double ToFeet(double metres)
        => UnitUtils.ConvertToInternalUnits(metres, UnitTypeId.Meters);

    /// <summary>
    /// Converts a raw Valor numeric value to metres based on its unit string.
    /// </summary>
    private static double ConvertToMetres(double value, string? unit)
        => unit?.ToLowerInvariant() switch
        {
            "cm" => value / 100.0,
            "mm" => value / 1000.0,
            _    => value,   // assumed metres
        };
}
