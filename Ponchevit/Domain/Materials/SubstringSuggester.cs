using System;
using System.Collections.Generic;
using System.Linq;
using Ponchevit.Domain.Model;

namespace Ponchevit.Domain.Materials;

/// <summary>
/// Suggests Covenin material value IDs from a Revit material name using
/// substring and partial-word matching. Used by the Mapeo de Materiales UI
/// to pre-populate the suggestion column.
/// </summary>
public static class SubstringSuggester
{
    public static IReadOnlyList<string> Suggest(
        string revitMaterialName,
        IEnumerable<Valor> coveninMaterialValues)
    {
        if (string.IsNullOrWhiteSpace(revitMaterialName))
            return Array.Empty<string>();

        var lower = revitMaterialName.ToLowerInvariant();

        return coveninMaterialValues
            .Where(v => IsMatch(lower, v.DescripcionUi.ToLowerInvariant()))
            .Select(v => v.IdValor)
            .ToList();
    }

    private static bool IsMatch(string revitLower, string coveninLower)
    {
        if (revitLower.Contains(coveninLower) || coveninLower.Contains(revitLower))
            return true;

        var revitWords = revitLower.Split([' ', '-', '_', '/'], StringSplitOptions.RemoveEmptyEntries);
        var coveninWords = coveninLower.Split([' ', '-', '_', '/'], StringSplitOptions.RemoveEmptyEntries);

        return revitWords.Any(rw =>
            rw.Length > 3 && coveninWords.Any(cw => cw.Contains(rw) || rw.Contains(cw)));
    }
}
