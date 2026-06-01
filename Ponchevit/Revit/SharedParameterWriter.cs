using System.Collections.Generic;
using Autodesk.Revit.DB;
using Ponchevit.Domain.Model;
using Ponchevit.Revit.SharedParameters;

namespace Ponchevit.Revit;

/// <summary>
/// Writes the 4 COVENIN shared parameters onto a Revit element.
/// Must be called inside a caller-supplied Transaction.
/// The extras dictionary is reserved for Post-MVP (see ADR 2026-05-31 — Extras dictionary);
/// passing a non-empty dict throws NotImplementedException.
/// </summary>
public static class SharedParameterWriter
{
    public static void Write(
        Element element,
        CodigoCovenin codigo,
        string capitulo,
        string subcapitulo,
        string seccion,
        IReadOnlyDictionary<string, string>? extras = null)
    {
        if (extras is { Count: > 0 })
            throw new System.NotImplementedException(
                "Extras dictionary is reserved for Post-MVP. See ADR 2026-05-31.");

        SetParam(element, CoveninParameters.CapituloName,       capitulo);
        SetParam(element, CoveninParameters.SubcapituloName,    subcapitulo);
        SetParam(element, CoveninParameters.SeccionName,        seccion);
        SetParam(element, CoveninParameters.CodigoCompletoName, codigo.Value);
    }

    private static void SetParam(Element element, string paramName, string value)
    {
        Parameter? param = element.LookupParameter(paramName);
        if (param == null || param.IsReadOnly)
            return;
        param.Set(value);
    }
}
