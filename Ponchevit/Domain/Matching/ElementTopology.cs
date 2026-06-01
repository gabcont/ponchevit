using System.Collections.Generic;

namespace Ponchevit.Domain.Matching;

/// <summary>
/// Pure-C# snapshot of a Revit element's structural topology and dimensions.
/// Produced by Revit/ElementTopologyReader; consumed by IElementMatcher implementations
/// (Phase 5) and by the Asignar prefill logic.
/// Dimension values are in Revit internal units (decimal feet).
/// </summary>
public sealed record ElementTopology(
    string Category,
    IReadOnlyList<MaterialLayer> Layers,
    IReadOnlyDictionary<string, double> Dimensions
);

/// <summary>
/// One layer from a Revit compound structure: the original Revit material name,
/// the mapped Covenin value ID (null when unmapped), and thickness in decimal feet.
/// </summary>
public sealed record MaterialLayer(
    string RevitMaterialName,
    string? CoveninMaterialValueId,
    double ThicknessFeet
);
