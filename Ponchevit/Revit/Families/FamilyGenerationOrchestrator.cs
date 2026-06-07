using System;
using Autodesk.Revit.DB;
using Ponchevit.Infrastructure;
using Ponchevit.Revit.SharedParameters;

namespace Ponchevit.Revit.Families;

/// <summary>
/// Orchestrates the two-step element generation flow on Revit's main thread:
///   1. Bind COVENIN shared parameters (own transaction, idempotent).
///   2. Generate the element via the supplied <see cref="IFamilyGenerator"/> (own transaction).
///
/// This class lives in Revit/ so the VM (Ui/) never touches Transaction or any other
/// RevitAPI type directly. The VM calls PostExternalEvent with a delegate that invokes
/// <see cref="Generate"/>; the orchestrator owns both transactions.
///
/// See ADR 2026-06-06 — Transaction-orchestration location.
/// </summary>
public sealed class FamilyGenerationOrchestrator
{
    private readonly ILog _log;

    public FamilyGenerationOrchestrator(ILog log)
    {
        _log = log ?? throw new ArgumentNullException(nameof(log));
    }

    /// <summary>
    /// Runs on Revit's main thread (inside a PostExternalEvent callback).
    /// </summary>
    /// <param name="doc">The active Revit document.</param>
    /// <param name="generator">The generator that knows how to create the element.</param>
    /// <param name="input">Assembled input including the 10-digit COVENIN code.</param>
    public void Generate(Document doc, IFamilyGenerator generator, GeneratorInput input)
    {
        if (doc       == null) throw new ArgumentNullException(nameof(doc));
        if (generator == null) throw new ArgumentNullException(nameof(generator));
        if (input     == null) throw new ArgumentNullException(nameof(input));

        // Step 1 — Bind shared parameters.
        // EnsureBoundToProject opens and commits its own Transaction internally.
        // It must complete before the creation transaction below, so Revit can see
        // the newly-bound parameters during element creation.
        CoveninParameters.EnsureBoundToProject(doc);

        // Step 2 — Create the element.
        // Generators expect to be called inside an already-open transaction.
        using var t = new Transaction(doc, "Agregar Familia COVENIN");
        t.Start();
        generator.Generate(doc, input);
        t.Commit();

        _log.Info($"FamilyGenerationOrchestrator: generated element with code {input.Codigo.Value}.");
    }
}
