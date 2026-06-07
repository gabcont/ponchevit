using System;
using System.Collections.Generic;
using System.Linq;
using Autodesk.Revit.DB;
using Ponchevit.Data;
using Ponchevit.Domain.Matching;
using Ponchevit.Domain.Model;
using Ponchevit.Infrastructure;

namespace Ponchevit.Revit.Families;

/// <summary>
/// Recognizes wall topology and suggests COVENIN DAG parameter values for prefill.
/// Hard-codes the known structural path for walls: E4 → Albañilería → Construcción de paredes.
/// Detects MATERIAL from compound layer mapping and ESPESOR from wall width (in cm).
/// Qualitative columns (ACABADO) are always returned as Undetectable by design.
/// </summary>
public sealed class MuroRecognizer : IElementRecognizer
{
    // ── COVENIN structural constants for walls ───────────────────────────────
    private const string WallSeccionValorId       = "VAL_00002";   // Construcción de paredes
    private const string MaterialColumnaId        = "COL_005";
    private const string AcabadoColumnaId         = "COL_006";
    private const string DigitosVacantesColumnaId = "COL_007";
    private const string DigitosVacantesValorId   = "VAL_00017";   // Sin Especificar
    private const string EspesorColumnaId         = "COL_009";
    private const double FeetToCm                 = 30.48;
    private const double EspesorToleranceCm       = 0.5;

    private readonly ICoveninRulesRepository _rulesRepo;
    private readonly ILog _log;

    public MuroRecognizer(ICoveninRulesRepository rulesRepo, ILog log)
    {
        _rulesRepo = rulesRepo ?? throw new ArgumentNullException(nameof(rulesRepo));
        _log       = log       ?? throw new ArgumentNullException(nameof(log));
    }

    public BuiltInCategory SupportedCategory => BuiltInCategory.OST_Walls;

    public bool CanRecognize(ElementTopology topology)
        => topology?.BuiltInCategoryId == (int)BuiltInCategory.OST_Walls;

    public PrefillResult Recognize(ElementTopology topology)
    {
        try
        {
            if (topology == null)
                return PrefillResult.Empty;

            var (capConexion, subCapConexion, secConexion, unConexion) = ResolveStructuralPath();

            if (capConexion == null || subCapConexion == null || secConexion == null || unConexion == null)
            {
                _log.Warn("MuroRecognizer: could not resolve structural path from covenin.db for walls.");
                return PrefillResult.Empty;
            }

            var entries = new Dictionary<string, PrefillEntry>
            {
                // COL_001 CAPITULO: IdValorAsociado is null in the DB, so we match by IdConexion only.
                [capConexion.IdColumna]    = new PrefillEntry(PrefillState.AutoFilled, null, capConexion.IdConexion),

                // COL_002 SUB-CAPITULO: Albañilería (VAL_00001), but two connections share this valor
                // (CON_000002 and CON_000714). We disambiguate by IdConexion.
                [subCapConexion.IdColumna] = new PrefillEntry(PrefillState.AutoFilled, subCapConexion.IdValorAsociado, subCapConexion.IdConexion),

                // COL_003 ACTIVIDAD: Construcción de paredes (VAL_00002).
                [secConexion.IdColumna]    = new PrefillEntry(PrefillState.AutoFilled, secConexion.IdValorAsociado, secConexion.IdConexion),

                // COL_004 UN.: m2 (VAL_00003) — single child, always this connection.
                [unConexion.IdColumna]     = new PrefillEntry(PrefillState.AutoFilled, unConexion.IdValorAsociado, unConexion.IdConexion),

                // COL_005 MATERIAL: detected from compound layer mapping.
                [MaterialColumnaId]        = RecognizeMaterial(topology),

                // COL_006 ACABADO: qualitative — user must choose.
                [AcabadoColumnaId]         = new PrefillEntry(PrefillState.Undetectable, null),

                // COL_007 DÍGITOS VACANTES: single option, always "Sin Especificar".
                [DigitosVacantesColumnaId] = new PrefillEntry(PrefillState.AutoFilled, DigitosVacantesValorId),

                // COL_009 ESPESOR: detected from wall width in cm.
                [EspesorColumnaId]         = RecognizeEspesor(topology),
            };

            return new PrefillResult(entries);
        }
        catch (Exception ex)
        {
            _log.Error("MuroRecognizer.Recognize failed.", ex);
            return PrefillResult.Empty;
        }
    }

    // ── Structural path resolution ───────────────────────────────────────────

    /// <summary>
    /// Walks the covenin.db DAG upward from the known wall sección connection
    /// to find capítulo, sub-capítulo, and downward to find the UN. connection.
    /// Returns all-null tuple if any node is missing.
    /// </summary>
    private (Conexion? cap, Conexion? subCap, Conexion? sec, Conexion? un) ResolveStructuralPath()
    {
        // Depth 3: Construcción de paredes (the one node we know by valor ID).
        var secConexion = _rulesRepo.GetConexionesByValorId(WallSeccionValorId).FirstOrDefault();
        if (secConexion == null) return (null, null, null, null);

        // Depth 2: walk up to sub-capítulo.
        if (secConexion.ParentId == null) return (null, null, null, null);
        var subCapConexion = _rulesRepo.GetConexionById(secConexion.ParentId);
        if (subCapConexion == null) return (null, null, null, null);

        // Depth 1: walk up to capítulo.
        if (subCapConexion.ParentId == null) return (null, null, null, null);
        var capConexion = _rulesRepo.GetConexionById(subCapConexion.ParentId);
        if (capConexion == null) return (null, null, null, null);

        // Depth 4: walk down to UN. (single child of secConexion).
        var unConexion = _rulesRepo.GetConexionesByParent(secConexion.IdConexion).FirstOrDefault();
        if (unConexion == null) return (null, null, null, null);

        return (capConexion, subCapConexion, secConexion, unConexion);
    }

    // ── Column-level recognition ─────────────────────────────────────────────

    private PrefillEntry RecognizeMaterial(ElementTopology topology)
    {
        var layerValueIds = topology.Layers
            .Where(l => l.CoveninMaterialValueId != null)
            .Select(l => l.CoveninMaterialValueId!)
            .ToHashSet(StringComparer.OrdinalIgnoreCase);

        if (layerValueIds.Count == 0)
            return new PrefillEntry(PrefillState.Undetectable, null);

        var matches = _rulesRepo.GetValores()
            .Where(v => string.Equals(v.IdColumna, MaterialColumnaId, StringComparison.OrdinalIgnoreCase)
                        && layerValueIds.Contains(v.IdValor))
            .ToList();

        return matches.Count switch
        {
            1 => new PrefillEntry(PrefillState.AutoFilled, matches[0].IdValor),
            > 1 => new PrefillEntry(PrefillState.Ambiguous, null),
            _ => new PrefillEntry(PrefillState.Undetectable, null)
        };
    }

    private PrefillEntry RecognizeEspesor(ElementTopology topology)
    {
        if (!topology.Dimensions.TryGetValue("Espesor", out double rawFeet))
            return new PrefillEntry(PrefillState.Undetectable, null);

        double valueCm = rawFeet * FeetToCm;

        var matches = _rulesRepo.GetValores()
            .Where(v => string.Equals(v.IdColumna, EspesorColumnaId, StringComparison.OrdinalIgnoreCase)
                        && v.NumMin.HasValue
                        && v.NumMax.HasValue
                        && v.NumMin.Value - EspesorToleranceCm <= valueCm
                        && valueCm <= v.NumMax.Value + EspesorToleranceCm)
            .ToList();

        return matches.Count switch
        {
            1 => new PrefillEntry(PrefillState.AutoFilled, matches[0].IdValor),
            > 1 => new PrefillEntry(PrefillState.Ambiguous, null),
            _ => new PrefillEntry(PrefillState.Undetectable, null)
        };
    }
}
