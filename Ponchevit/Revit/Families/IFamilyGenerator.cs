using System.Collections.Generic;
using Autodesk.Revit.DB;
using Ponchevit.Domain.Model;

namespace Ponchevit.Revit.Families;

/// <summary>
/// Strategy interface for generating a code-compliant Revit family instance.
/// One implementation per supported element category (Muro for MVP).
/// Register implementations in Composition/Services.cs.
/// </summary>
public interface IFamilyGenerator
{
    BuiltInCategory SupportedCategory { get; }

    /// <summary>
    /// Creates the element inside the given document. Must be called inside a transaction.
    /// </summary>
    void Generate(Document doc, GeneratorInput input);
}

/// <summary>
/// Encapsulates all the information needed to generate a code-compliant element.
/// CoveninValues maps IdColumna → IdValor (selected in the central panel).
/// RevitParameters holds Revit-specific values like location curve, level, height.
/// </summary>
public sealed record GeneratorInput(
    CodigoCovenin Codigo,
    string Capitulo,
    string? Subcapitulo,
    string? Seccion,
    IReadOnlyDictionary<string, string> CoveninValues,
    IReadOnlyDictionary<string, object> RevitParameters
);
