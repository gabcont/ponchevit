using System;
using Autodesk.Revit.DB;
using Ponchevit.Domain.Aliases;
using Ponchevit.Domain.Model;
using Ponchevit.Infrastructure;
using Ponchevit.Revit.SharedParameters;

namespace Ponchevit.Revit.Families;

/// <summary>
/// Orchestrates writing 4 COVENIN shared parameters onto an existing element.
/// Owns the Manual transaction; must be called from Revit's main thread
/// (inside a PostExternalEvent callback).
/// </summary>
public sealed class AssignCodeOrchestrator
{
    private readonly ILog _log;
    private readonly IAliasResolver _aliasResolver;

    public AssignCodeOrchestrator(ILog log, IAliasResolver aliasResolver)
    {
        _log           = log           ?? throw new ArgumentNullException(nameof(log));
        _aliasResolver = aliasResolver ?? throw new ArgumentNullException(nameof(aliasResolver));
    }

    /// <summary>
    /// Overload that accepts the element's <see cref="ElementId.Value"/> as a <see langword="long"/>,
    /// so callers in layers that must not reference RevitAPI can invoke this without constructing
    /// an <see cref="ElementId"/> themselves.
    /// </summary>
    public void Assign(Document doc, long elementIdValue, AssignInput input)
        => Assign(doc, new ElementId(elementIdValue), input);

    public void Assign(Document doc, ElementId elementId, AssignInput input)
    {
        if (doc       == null) throw new ArgumentNullException(nameof(doc));
        if (elementId == null) throw new ArgumentNullException(nameof(elementId));
        if (input     == null) throw new ArgumentNullException(nameof(input));

        var resolvedCode = new CodigoCovenin(_aliasResolver.Resolve(input.Codigo.Value));

        CoveninParameters.EnsureBoundToProject(doc);

        using var t = new Transaction(doc, "Asignar Código COVENIN");
        t.Start();

        var element = doc.GetElement(elementId);
        if (element == null)
            throw new InvalidOperationException(
                $"Element {elementId.Value} not found in the document.");

        // COVENIN params are type-bound; write on the element's type, not the instance.
        var typeId = element.GetTypeId();
        var targetElement = (typeId != null && typeId != ElementId.InvalidElementId)
            ? (doc.GetElement(typeId) ?? element)
            : element;

        SharedParameterWriter.Write(
            targetElement, resolvedCode,
            input.Capitulo, input.Subcapitulo, input.Seccion,
            extras: null);

        t.Commit();
        _log.Info($"AssignCodeOrchestrator: assigned code {resolvedCode.Value} to element {elementId.Value}.");
    }
}
