using System.Collections.Generic;
using Autodesk.Revit.DB;
using Ponchevit.Domain.Model;

namespace Ponchevit.Revit.Families;

/// <summary>
/// Strategy interface for generating a code-compliant Revit family element.
/// One implementation per supported element category (Muro for MVP).
///
/// Two concepts are deliberately distinct:
///   <see cref="IsGenerable"/>  — a generator module exists and can act on this code prefix.
///   CanBeConstructed (on PartidaConstructibilityResolver) — the COVENIN DAG has a valid
///   root-to-leaf path producing this code (does not imply a generator exists).
///
/// To add support for a new element type, implement this interface and register the
/// instance in the element-module registry in Composition/Services.cs.
/// </summary>
public interface IFamilyGenerator
{
    BuiltInCategory SupportedCategory { get; }

    /// <summary>
    /// Returns true when this generator can produce an element for the given code prefix.
    /// Implementations must handle null/empty input gracefully (return false).
    /// Used by the VM (as the <em>IsGenerable</em> check) to enable/disable tree nodes
    /// without referencing BuiltInCategory directly.
    /// </summary>
    bool IsGenerable(string? codigoPrefix);

    /// <summary>
    /// Creates or configures the element inside the given document.
    /// Must be called inside a transaction (owned by FamilyGenerationOrchestrator).
    /// </summary>
    void Generate(Document doc, GeneratorInput input);
}

/// <summary>
/// Encapsulates all the information needed to generate a code-compliant element.
/// CoveninValues maps IdColumna → IdValor (selected in the central panel).
/// SelectedValores maps IdColumna → Valor (full semantic value objects).
/// NumericValues maps IdColumna → double value in metres (for dimensional columns).
/// </summary>
public sealed record GeneratorInput(
    CodigoCovenin Codigo,
    string Capitulo,
    string? Subcapitulo,
    string? Seccion,
    string Descripcion,
    IReadOnlyDictionary<string, string> CoveninValues,
    IReadOnlyDictionary<string, Ponchevit.Domain.Model.Valor> SelectedValores,
    IReadOnlyDictionary<string, double> NumericValues
);
