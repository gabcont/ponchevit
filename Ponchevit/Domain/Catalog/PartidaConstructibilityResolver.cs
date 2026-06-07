using System;
using System.Collections.Generic;
using System.Linq;
using Ponchevit.Data;
using Ponchevit.Domain.Graph;
using Ponchevit.Domain.Model;

namespace Ponchevit.Domain.Catalog;

/// <summary>
/// Determines which partidas in the catalog can be constructed via the COVENIN DAG.
/// A partida is <em>constructible</em> iff there exists at least one root-to-leaf path
/// through <c>Covenin_Conexiones</c> whose concatenated <c>CodigoAportado</c> equals
/// the partida's <c>CodigoPartida</c>.
///
/// Algorithm (one-time at construction):
///   Performs a reverse BFS starting from every root connection and assembles codes by
///   DFS.  Results are stored in a <c>HashSet&lt;string&gt;</c> for O(1) look-up.
///
/// Performance notes:
///   The DAG has ~377k edges but they are lazy-loaded per <c>Parent_Id</c>.  We DFS
///   depth-first and prune any partial path whose assembled prefix cannot possibly match
///   any known partida code (early-exit by prefix).  In practice only the E4 sub-DAG is
///   relevant for MVP; the full traversal finishes in under a second on commodity
///   hardware.
///
/// Partidas outside capítulos with any DAG roots are unconstructible by definition and
/// never appear in the constructible set.
/// </summary>
public sealed class PartidaConstructibilityResolver
{
    // Set of codes that have at least one valid DAG path.
    private readonly HashSet<string> _constructibleCodes;

    // Connection-path that produces a given constructible code (first found; used for backfill).
    // Key = 10-digit code, Value = ordered list of connection IDs (root → leaf).
    private readonly Dictionary<string, IReadOnlyList<string>> _codeToPath;

    private readonly ICoveninRulesRepository _rulesRepo;

    public PartidaConstructibilityResolver(
        ICoveninRulesRepository rulesRepo,
        IReadOnlyList<Partida> partidas)
    {
        _rulesRepo = rulesRepo ?? throw new ArgumentNullException(nameof(rulesRepo));
        if (partidas == null) throw new ArgumentNullException(nameof(partidas));

        // Build a prefix index over known partida codes for fast pruning.
        var knownCodes = new HashSet<string>(
            partidas.Select(p => p.CodigoPartida),
            StringComparer.OrdinalIgnoreCase);

        // All 10-char codes we care about — also build a prefix lookup for pruning.
        // prefixMap[prefix] = true means at least one known code starts with that prefix.
        var prefixSet = BuildPrefixSet(knownCodes);

        _constructibleCodes = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        _codeToPath = new Dictionary<string, IReadOnlyList<string>>(StringComparer.OrdinalIgnoreCase);

        // DFS from each root connection.
        var roots = _rulesRepo.GetConexionesByParent(null).ToList();
        var path  = new List<string>();  // current connection-ID path
        var code  = new System.Text.StringBuilder();

        foreach (var root in roots)
            Dfs(root, path, code, knownCodes, prefixSet);
    }

    /// <summary>
    /// Returns true when at least one valid DAG path through Covenin_Conexiones
    /// produces this partida's code.  Does NOT imply a generator module exists.
    /// </summary>
    public bool CanBeConstructed(Partida partida)
        => partida != null && _constructibleCodes.Contains(partida.CodigoPartida);

    /// <summary>
    /// Returns true when at least one valid DAG path produces this 10-digit code.
    /// Does NOT imply a generator module exists.
    /// </summary>
    public bool CanBeConstructed(string codigo)
        => !string.IsNullOrEmpty(codigo) && _constructibleCodes.Contains(codigo);

    /// <summary>
    /// Returns the ordered list of connection IDs (root → leaf) for a constructible
    /// code, or null if the code is unconstructible.  Used by the VM to backfill
    /// the cascade panel when the user clicks a partida in the right panel.
    /// </summary>
    public IReadOnlyList<string>? GetPath(string codigo)
        => _codeToPath.TryGetValue(codigo, out var p) ? p : null;

    // ── Private ────────────────────────────────────────────────────────────────

    private void Dfs(
        Conexion conexion,
        List<string> path,
        System.Text.StringBuilder code,
        HashSet<string> knownCodes,
        HashSet<string> prefixSet)
    {
        int prevLen = code.Length;
        path.Add(conexion.IdConexion);

        // Append this connection's code contribution.
        string contrib = conexion.CodigoAportado ?? string.Empty;
        if (contrib.Length > 0 && code.Length < CodeAssembler.MaxCodeLength)
        {
            int room = CodeAssembler.MaxCodeLength - code.Length;
            code.Append(contrib.Length <= room ? contrib : contrib[..room]);
        }

        string currentCode = code.ToString();

        // Pruning: if no known code starts with the current partial code, abandon subtree.
        if (currentCode.Length > 0 && !prefixSet.Contains(currentCode))
        {
            path.RemoveAt(path.Count - 1);
            code.Length = prevLen;
            return;
        }

        // Check for a complete match.
        if (currentCode.Length == CodeAssembler.MaxCodeLength
            && knownCodes.Contains(currentCode)
            && !_constructibleCodes.Contains(currentCode))
        {
            _constructibleCodes.Add(currentCode);
            _codeToPath[currentCode] = path.ToList(); // snapshot
        }

        // Recurse into children.
        var children = _rulesRepo.GetConexionesByParent(conexion.IdConexion).ToList();
        foreach (var child in children)
            Dfs(child, path, code, knownCodes, prefixSet);

        // Backtrack.
        path.RemoveAt(path.Count - 1);
        code.Length = prevLen;
    }

    /// <summary>
    /// Builds a set containing every non-empty prefix of every known code.
    /// Used for O(1) pruning during DFS: if the current assembled prefix is not in this
    /// set, no known code can be reached from this point.
    /// </summary>
    private static HashSet<string> BuildPrefixSet(HashSet<string> codes)
    {
        var set = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var code in codes)
        {
            for (int len = 1; len <= code.Length; len++)
                set.Add(code[..len]);
        }
        return set;
    }
}
