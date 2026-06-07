using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using Ponchevit.Domain.Catalog;
using Ponchevit.Domain.Model;

namespace Ponchevit.Ui.Common;

/// <summary>
/// Observable collection of Partidas that applies a fast in-memory filter driven by
/// Capítulo / Subcapítulo / Sección selection state and an optional DAG-derived code
/// prefix mask. Delegates predicate logic entirely to <see cref="PartidaFilter"/> — no
/// filtering logic lives here.
///
/// Usage:
///   var col = new FilteredPartidaCollection(catalog.GetPartidas());
///   col.ApplyFilter(capituloCodigo: "E4", subcapituloCodigo: "E41");
///   // Or, with a cascade-derived mask:
///   col.ApplyFilter(codePrefix: "E411XXXXXX");
///   // Bind col.Items to a ListBox/DataGrid.
///
/// Thread affinity: all calls must come from the UI thread (WPF binding requirement).
/// </summary>
public sealed class FilteredPartidaCollection
{
    private readonly IReadOnlyList<Partida> _allPartidas;
    private readonly PartidaFilter _filter = new();

    // Exposed as a writable ObservableCollection so WPF change-tracking works.
    public ObservableCollection<Partida> Items { get; } = new();

    // ── Current filter criteria ──────────────────────────────────────────────
    private string? _capituloCodigo;
    private string? _subcapituloCodigo;
    private string? _seccionCodigo;
    private string? _codePrefix;

    public FilteredPartidaCollection(IReadOnlyList<Partida> allPartidas)
    {
        _allPartidas = allPartidas ?? throw new ArgumentNullException(nameof(allPartidas));
        Refresh();
    }

    /// <summary>
    /// Replaces the filter criteria and synchronously refreshes <see cref="Items"/>.
    /// Pass null to any parameter to clear that level. Never throws — always produces
    /// a valid (possibly empty) Items list.
    ///
    /// When <paramref name="codePrefix"/> is supplied it takes priority over the
    /// hierarchy codes and uses wildcard matching (X = any character at that position).
    /// </summary>
    public void ApplyFilter(
        string? capituloCodigo    = null,
        string? subcapituloCodigo = null,
        string? seccionCodigo     = null,
        string? codePrefix        = null)
    {
        _capituloCodigo    = capituloCodigo;
        _subcapituloCodigo = subcapituloCodigo;
        _seccionCodigo     = seccionCodigo;
        _codePrefix        = codePrefix;
        Refresh();
    }

    /// <summary>
    /// Clears all filter criteria, restoring the full set. Equivalent to
    /// <c>ApplyFilter(null, null, null, null)</c>.
    /// </summary>
    public void ClearFilter() => ApplyFilter();

    // ── Private helpers ──────────────────────────────────────────────────────

    private void Refresh()
    {
        var filtered = _filter
            .Filter(_allPartidas, _capituloCodigo, _subcapituloCodigo, _seccionCodigo, _codePrefix)
            .ToList();

        // Rebuild Items in-place so WPF collections stay bound. For the scale involved
        // (~2081 partidas) a full Clear + re-add is fast enough; no diff needed.
        Items.Clear();
        foreach (var p in filtered)
            Items.Add(p);
    }
}
