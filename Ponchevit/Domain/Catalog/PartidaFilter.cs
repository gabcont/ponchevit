namespace Ponchevit.Domain.Catalog;

using System;
using System.Collections.Generic;
using System.Linq;
using Ponchevit.Domain.Model;

/// <summary>
/// Pure predicate filter for partidas. Delegates filtering to hierarchy codes
/// or an optional DAG-derived code prefix mask (X = wildcard).
/// </summary>
public class PartidaFilter
{
    /// <summary>
    /// Filters partidas by hierarchy codes and/or an optional code prefix mask.
    /// When <paramref name="codePrefix"/> is supplied it takes priority over the
    /// hierarchy codes and uses wildcard matching (X = any character at that position).
    /// </summary>
    public IEnumerable<Partida> Filter(
        IEnumerable<Partida> partidas,
        string? capituloCodigo    = null,
        string? subcapituloCodigo = null,
        string? seccionCodigo     = null,
        string? codePrefix        = null)
    {
        if (partidas == null) return Enumerable.Empty<Partida>();

        // Code prefix mask takes priority (e.g. "E411XXXXXX").
        if (!string.IsNullOrEmpty(codePrefix))
            return partidas.Where(p => MatchesPrefix(p.CodigoPartida, codePrefix));

        // In COVENIN E4, codes are hierarchical: Seccion starts with Subcapitulo, which starts with Capitulo.
        // We filter by the most specific prefix provided.
        string? effectivePrefix = seccionCodigo ?? subcapituloCodigo ?? capituloCodigo;

        if (string.IsNullOrWhiteSpace(effectivePrefix))
            return partidas;

        return partidas.Where(p => p.CodigoPartida.StartsWith(effectivePrefix, StringComparison.OrdinalIgnoreCase));
    }

    /// <summary>
    /// Returns true when the code matches the prefix mask where 'X' is a wildcard
    /// matching any single character.
    /// </summary>
    private static bool MatchesPrefix(string code, string mask)
    {
        if (code.Length < mask.Length) return false;
        for (int i = 0; i < mask.Length; i++)
        {
            char m = char.ToUpperInvariant(mask[i]);
            if (m == 'X') continue;
            if (char.ToUpperInvariant(code[i]) != m) return false;
        }
        return true;
    }
}
