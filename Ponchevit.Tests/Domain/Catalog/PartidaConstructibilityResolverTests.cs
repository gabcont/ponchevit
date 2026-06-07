using System.Collections.Generic;
using System.Linq;
using Ponchevit.Data;
using Ponchevit.Domain.Catalog;
using Ponchevit.Domain.Model;
using Xunit;

namespace Ponchevit.Tests.Domain.Catalog;

/// <summary>
/// Tests for PartidaConstructibilityResolver.
///
/// The resolver determines whether a partida's 10-digit code can be assembled from a
/// root-to-leaf path through the COVENIN DAG (Covenin_Conexiones).
/// </summary>
public class PartidaConstructibilityResolverTests
{
    // ── Fake repository ───────────────────────────────────────────────────────

    /// <summary>
    /// Minimal in-memory ICoveninRulesRepository for testing.
    /// Builds a small DAG from a flat list of (id, parentId, codigoAportado) tuples.
    /// </summary>
    private sealed class FakeRulesRepo : ICoveninRulesRepository
    {
        private readonly List<Conexion> _all;

        public FakeRulesRepo(IEnumerable<(string id, string? parent, string codigo)> edges)
        {
            _all = edges
                .Select(e => new Conexion(e.id, e.parent, e.codigo, "col1", null))
                .ToList();
        }

        public IEnumerable<Columna> GetColumnas() => Enumerable.Empty<Columna>();
        public IEnumerable<Valor> GetValores() => Enumerable.Empty<Valor>();
        public Columna? GetColumna(string id) => null;
        public Valor? GetValor(string id) => null;

        public IEnumerable<Conexion> GetConexionesByParent(string? parentId)
            => _all.Where(c =>
                string.IsNullOrEmpty(parentId)
                    ? c.ParentId == null || c.ParentId == string.Empty
                    : c.ParentId == parentId);

        public Conexion? GetConexionById(string idConexion)
            => _all.FirstOrDefault(c => c.IdConexion == idConexion);

        public IEnumerable<Conexion> GetConexionesByValorId(string idValorAsociado)
            => _all.Where(c => c.IdValorAsociado == idValorAsociado);
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static Partida P(string code) =>
        new(code, "m2", "Test", "E4");

    // ── Constructible detection ───────────────────────────────────────────────

    [Fact]
    public void CanBeConstructed_WhenPathAssemblesExactCode_ReturnsTrue()
    {
        // DAG: root → A (E4) → B (11) → C (011) → D (001) → leaf (nothing)
        // Assembled: "E4" + "11" + "011" + "001" = "E411011001" (10 chars)
        var repo = new FakeRulesRepo(new[]
        {
            ("A", (string?)null, "E4"),
            ("B",           "A", "11"),
            ("C",           "B", "011"),
            ("D",           "C", "001"),
            ("E",           "D", "X"),  // extra char — trimmed to 10
        });

        var partidas = new List<Partida> { P("E411011001") };
        var resolver = new PartidaConstructibilityResolver(repo, partidas);

        Assert.True(resolver.CanBeConstructed(P("E411011001")));
    }

    [Fact]
    public void CanBeConstructed_WhenNoPathMatchesCode_ReturnsFalse()
    {
        // DAG only assembles "E411011001"; asking for "E411011002"
        var repo = new FakeRulesRepo(new[]
        {
            ("A", (string?)null, "E4"),
            ("B",           "A", "11"),
            ("C",           "B", "011"),
            ("D",           "C", "001"),
        });

        var partidas = new List<Partida> { P("E411011001"), P("E411011002") };
        var resolver = new PartidaConstructibilityResolver(repo, partidas);

        Assert.True(resolver.CanBeConstructed(P("E411011001")));
        Assert.False(resolver.CanBeConstructed(P("E411011002")));
    }

    [Fact]
    public void CanBeConstructed_PartidaOutsideAnyDagChapter_ReturnsFalse()
    {
        // DAG contains only E4xx paths; partida is under E5.
        var repo = new FakeRulesRepo(new[]
        {
            ("A", (string?)null, "E411011001"),
        });

        var e5Partida = P("E511011001");
        var resolver = new PartidaConstructibilityResolver(repo, new List<Partida> { e5Partida });

        Assert.False(resolver.CanBeConstructed(e5Partida));
    }

    [Fact]
    public void CanBeConstructed_EmptyDag_AllPartidas_ReturnsFalse()
    {
        var repo = new FakeRulesRepo(Enumerable.Empty<(string, string?, string)>());
        var partidas = new List<Partida> { P("E411011001"), P("E412011001") };
        var resolver = new PartidaConstructibilityResolver(repo, partidas);

        Assert.False(resolver.CanBeConstructed(P("E411011001")));
        Assert.False(resolver.CanBeConstructed(P("E412011001")));
    }

    [Fact]
    public void CanBeConstructed_EmptyPartidaList_NeverThrows()
    {
        var repo = new FakeRulesRepo(new[]
        {
            ("A", (string?)null, "E411011001"),
        });

        var ex = Record.Exception(
            () => new PartidaConstructibilityResolver(repo, new List<Partida>()));
        Assert.Null(ex);
    }

    // ── Path retrieval ────────────────────────────────────────────────────────

    [Fact]
    public void GetPath_ConstructibleCode_ReturnsConnectionIdSequence()
    {
        var repo = new FakeRulesRepo(new[]
        {
            ("A", (string?)null, "E4"),
            ("B",           "A", "11"),
            ("C",           "B", "011"),
            ("D",           "C", "001"),
        });

        var partidas = new List<Partida> { P("E411011001") };
        var resolver = new PartidaConstructibilityResolver(repo, partidas);

        var path = resolver.GetPath("E411011001");
        Assert.NotNull(path);
        Assert.Equal(new[] { "A", "B", "C", "D" }, path);
    }

    [Fact]
    public void GetPath_UnconstructibleCode_ReturnsNull()
    {
        var repo = new FakeRulesRepo(Enumerable.Empty<(string, string?, string)>());
        var resolver = new PartidaConstructibilityResolver(repo, new List<Partida>());

        Assert.Null(resolver.GetPath("E411011001"));
    }

    // ── Multiple constructible partidas ───────────────────────────────────────

    [Fact]
    public void Resolver_MultiplePartidas_DetectsEachIndependently()
    {
        // Two sibling leaves: D assembles E411011001, E assembles E411011002
        var repo = new FakeRulesRepo(new[]
        {
            ("A", (string?)null, "E4"),
            ("B",           "A", "11"),
            ("C",           "B", "011"),
            ("D",           "C", "001"),   // → E411011001
            ("E",           "C", "002"),   // → E411011002
        });

        var partidas = new List<Partida>
        {
            P("E411011001"),
            P("E411011002"),
            P("E411011003"),  // not in DAG
        };
        var resolver = new PartidaConstructibilityResolver(repo, partidas);

        Assert.True(resolver.CanBeConstructed(P("E411011001")));
        Assert.True(resolver.CanBeConstructed(P("E411011002")));
        Assert.False(resolver.CanBeConstructed(P("E411011003")));
    }

    // ── Pruning correctness ───────────────────────────────────────────────────

    [Fact]
    public void Resolver_DoesNotTraverseSubtreesWithNoKnownMatch()
    {
        // One branch leads to "ZZZZZZZZZZ" which is not in the known partidas.
        // The resolver must still return quickly (pruning) and not find that code.
        var repo = new FakeRulesRepo(new[]
        {
            ("A", (string?)null, "E411011001"),  // direct 10-char root
            ("B", (string?)null, "ZZZZZZZZZZ"),  // unknown — pruned immediately
        });

        var partidas = new List<Partida> { P("E411011001") };
        var resolver = new PartidaConstructibilityResolver(repo, partidas);

        Assert.True(resolver.CanBeConstructed("E411011001"));
        Assert.False(resolver.CanBeConstructed("ZZZZZZZZZZ"));
    }

    // ── Null / edge-case inputs ───────────────────────────────────────────────

    [Fact]
    public void CanBeConstructed_NullPartida_ReturnsFalse()
    {
        var repo = new FakeRulesRepo(Enumerable.Empty<(string, string?, string)>());
        var resolver = new PartidaConstructibilityResolver(repo, new List<Partida>());

        Assert.False(resolver.CanBeConstructed((Partida)null!));
    }

    [Fact]
    public void CanBeConstructed_NullOrEmptyCode_ReturnsFalse()
    {
        var repo = new FakeRulesRepo(Enumerable.Empty<(string, string?, string)>());
        var resolver = new PartidaConstructibilityResolver(repo, new List<Partida>());

        Assert.False(resolver.CanBeConstructed((string)null!));
        Assert.False(resolver.CanBeConstructed(string.Empty));
    }
}
