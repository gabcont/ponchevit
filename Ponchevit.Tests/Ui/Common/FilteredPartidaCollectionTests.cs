using System.Collections.Generic;
using System.Collections.Specialized;
using System.Linq;
using Ponchevit.Domain.Model;
using Ponchevit.Ui.Common;
using Xunit;

namespace Ponchevit.Tests.Ui.Common;

public class FilteredPartidaCollectionTests
{
    // ── Fixture data ─────────────────────────────────────────────────────────
    private static IReadOnlyList<Partida> SamplePartidas() => new List<Partida>
    {
        new("E411011001", "m2", "Muro ladrillo 15cm",  "E4"),
        new("E411011002", "m2", "Muro ladrillo 20cm",  "E4"),
        new("E412011001", "m2", "Muro bloque",         "E4"),
        new("E420011001", "m3", "Concreto simple",     "E4"),
        new("E511011001", "m",  "Tubería 2\"",          "E5"),
    };

    // ── Construction ─────────────────────────────────────────────────────────

    [Fact]
    public void Constructor_WithEmptyList_ProducesEmptyItems()
    {
        var col = new FilteredPartidaCollection(new List<Partida>());
        Assert.Empty(col.Items);
    }

    [Fact]
    public void Constructor_WithPartidas_ShowsAll()
    {
        var col = new FilteredPartidaCollection(SamplePartidas());
        Assert.Equal(5, col.Items.Count);
    }

    // ── ApplyFilter ───────────────────────────────────────────────────────────

    [Fact]
    public void ApplyFilter_ByCapitulo_NarrowsToMatchingPrefix()
    {
        var col = new FilteredPartidaCollection(SamplePartidas());
        col.ApplyFilter(capituloCodigo: "E4");
        Assert.Equal(4, col.Items.Count);
        Assert.All(col.Items, p => Assert.StartsWith("E4", p.CodigoPartida));
    }

    [Fact]
    public void ApplyFilter_BySubcapitulo_NarrowsFurther()
    {
        var col = new FilteredPartidaCollection(SamplePartidas());
        col.ApplyFilter(subcapituloCodigo: "E41");
        // E411* (2) + E412* (1) = 3
        Assert.Equal(3, col.Items.Count);
        Assert.All(col.Items, p => Assert.StartsWith("E41", p.CodigoPartida));
    }

    [Fact]
    public void ApplyFilter_BySeccion_MostSpecificWins()
    {
        var col = new FilteredPartidaCollection(SamplePartidas());
        col.ApplyFilter(capituloCodigo: "E4", subcapituloCodigo: "E41", seccionCodigo: "E411");
        // SeccionCodigo is most specific: E411* (2)
        Assert.Equal(2, col.Items.Count);
        Assert.All(col.Items, p => Assert.StartsWith("E411", p.CodigoPartida));
    }

    [Fact]
    public void ApplyFilter_OnlySeccion_Works()
    {
        var col = new FilteredPartidaCollection(SamplePartidas());
        col.ApplyFilter(seccionCodigo: "E412");
        Assert.Single(col.Items);
        Assert.Equal("E412011001", col.Items[0].CodigoPartida);
    }

    [Fact]
    public void ApplyFilter_NullArgs_RestoresAll()
    {
        var col = new FilteredPartidaCollection(SamplePartidas());
        col.ApplyFilter(capituloCodigo: "E5");
        Assert.Single(col.Items);

        col.ApplyFilter(); // clear
        Assert.Equal(5, col.Items.Count);
    }

    [Fact]
    public void ApplyFilter_NoMatch_ReturnsEmptyItems()
    {
        var col = new FilteredPartidaCollection(SamplePartidas());
        col.ApplyFilter(capituloCodigo: "X9");
        Assert.Empty(col.Items);
    }

    // ── ClearFilter ───────────────────────────────────────────────────────────

    [Fact]
    public void ClearFilter_AfterNarrowing_RestoresAll()
    {
        var col = new FilteredPartidaCollection(SamplePartidas());
        col.ApplyFilter(capituloCodigo: "E5");
        col.ClearFilter();
        Assert.Equal(5, col.Items.Count);
    }

    // ── Change notification ───────────────────────────────────────────────────

    [Fact]
    public void ApplyFilter_RaisesCollectionChangedNotification()
    {
        var col     = new FilteredPartidaCollection(SamplePartidas());
        int raised  = 0;
        col.Items.CollectionChanged += (_, e) =>
        {
            if (e.Action == NotifyCollectionChangedAction.Reset ||
                e.Action == NotifyCollectionChangedAction.Add   ||
                e.Action == NotifyCollectionChangedAction.Remove)
                raised++;
        };

        col.ApplyFilter(capituloCodigo: "E4");
        Assert.True(raised > 0, "CollectionChanged should fire at least once during filter.");
    }

    // ── Edge cases ────────────────────────────────────────────────────────────

    [Fact]
    public void ApplyFilter_CalledMultipleTimes_DoesNotGrow()
    {
        var col = new FilteredPartidaCollection(SamplePartidas());
        col.ApplyFilter(capituloCodigo: "E4");
        col.ApplyFilter(capituloCodigo: "E4");
        col.ApplyFilter(capituloCodigo: "E4");
        Assert.Equal(4, col.Items.Count);
    }

    [Fact]
    public void ApplyFilter_WithNoRulesForCapitulo_ReturnsEmptyAndDoesNotThrow()
    {
        // Simulates the "Sin reglas COVENIN" scenario: the filter still operates
        // on the flat catalog; it will just return an empty set for unknown prefixes.
        var col = new FilteredPartidaCollection(SamplePartidas());
        var ex  = Record.Exception(() => col.ApplyFilter(capituloCodigo: "NONEXISTENT"));
        Assert.Null(ex);
        Assert.Empty(col.Items);
    }

    [Fact]
    public void Constructor_NeverThrowsOnEmptyOrNullSelection()
    {
        var ex = Record.Exception(() =>
        {
            var col = new FilteredPartidaCollection(new List<Partida>());
            col.ApplyFilter();
            col.ClearFilter();
        });
        Assert.Null(ex);
    }

    // ── Ordering stability ────────────────────────────────────────────────────

    [Fact]
    public void ApplyFilter_PreservesSourceOrder()
    {
        var col = new FilteredPartidaCollection(SamplePartidas());
        col.ApplyFilter(capituloCodigo: "E4");
        var expected = SamplePartidas()
            .Where(p => p.CodigoPartida.StartsWith("E4"))
            .Select(p => p.CodigoPartida)
            .ToList();
        var actual = col.Items.Select(p => p.CodigoPartida).ToList();
        Assert.Equal(expected, actual);
    }

    // ── Code-prefix mask filtering ────────────────────────────────────────────

    [Fact]
    public void ApplyFilter_WithCodePrefix_AllWildcards_ShowsAll()
    {
        var col = new FilteredPartidaCollection(SamplePartidas());
        col.ApplyFilter(codePrefix: "XXXXXXXXXX");
        Assert.Equal(5, col.Items.Count);
    }

    [Fact]
    public void ApplyFilter_WithCodePrefix_NarrowsToMatchingPartidas()
    {
        var col = new FilteredPartidaCollection(SamplePartidas());
        col.ApplyFilter(codePrefix: "E411XXXXXX");
        Assert.Equal(2, col.Items.Count);
        Assert.All(col.Items, p => Assert.StartsWith("E411", p.CodigoPartida));
    }

    [Fact]
    public void ApplyFilter_WithCodePrefix_TakesPriorityOverHierarchyCodes()
    {
        var col = new FilteredPartidaCollection(SamplePartidas());
        // Hierarchy says E5, but mask says E411 — mask wins.
        col.ApplyFilter(capituloCodigo: "E5", codePrefix: "E411XXXXXX");
        Assert.Equal(2, col.Items.Count);
        Assert.All(col.Items, p => Assert.StartsWith("E411", p.CodigoPartida));
    }

    [Fact]
    public void ApplyFilter_WithFullCodePrefix_ReturnsSinglePartida()
    {
        var col = new FilteredPartidaCollection(SamplePartidas());
        col.ApplyFilter(codePrefix: "E411011001");
        Assert.Single(col.Items);
        Assert.Equal("E411011001", col.Items[0].CodigoPartida);
    }

    [Fact]
    public void ApplyFilter_ClearingCodePrefix_RestoresPriorHierarchyFilter()
    {
        var col = new FilteredPartidaCollection(SamplePartidas());
        col.ApplyFilter(codePrefix: "E411XXXXXX");
        Assert.Equal(2, col.Items.Count);

        // Clear codePrefix; re-apply hierarchy filter.
        col.ApplyFilter(capituloCodigo: "E5");
        Assert.Single(col.Items);
        Assert.Equal("E511011001", col.Items[0].CodigoPartida);
    }

    // ── Greyed-out semantics (constructibility) ───────────────────────────────
    //
    // FilteredPartidaCollection itself does not know about constructibility —
    // that annotation is added by the VM via PartidaDisplayItem. These tests
    // verify the underlying filter still operates on Partida.CodigoPartida
    // so the VM can layer constructibility on top without affecting filter results.

    [Fact]
    public void ApplyFilter_DoesNotExcludeUnconstructiblePartidas()
    {
        // The filter should return unconstructible partidas (greying is a VM concern).
        // All five SamplePartidas() should appear when no filter is active.
        var col = new FilteredPartidaCollection(SamplePartidas());
        col.ClearFilter();
        Assert.Equal(5, col.Items.Count);
    }
}
