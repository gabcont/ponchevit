using System.Collections.Generic;
using Ponchevit.Data;
using Ponchevit.Domain.Materials;
using Ponchevit.Domain.Model;
using Xunit;

namespace Ponchevit.Tests.Domain.Materials;

// ── In-memory fake ────────────────────────────────────────────────────────────

internal sealed class InMemoryMaterialMappingRepository : IMaterialMappingRepository
{
    private readonly Dictionary<string, string> _map = new();

    public IReadOnlyDictionary<string, string> GetAll() => _map;
    public void Set(string name, string id) => _map[name] = id;
    public void Remove(string name) => _map.Remove(name);
    public void Clear() => _map.Clear();
}

// ── MaterialMappingResolver tests ─────────────────────────────────────────────

public class MaterialMappingResolverTests
{
    [Fact]
    public void Resolve_KnownMaterial_ReturnsCoveninId()
    {
        var repo = new InMemoryMaterialMappingRepository();
        repo.Set("Bloque de arcilla", "V001");
        var resolver = new MaterialMappingResolver(repo);

        Assert.Equal("V001", resolver.Resolve("Bloque de arcilla"));
    }

    [Fact]
    public void Resolve_UnknownMaterial_ReturnsNull()
    {
        var resolver = new MaterialMappingResolver(new InMemoryMaterialMappingRepository());
        Assert.Null(resolver.Resolve("Acero inoxidable"));
    }

    [Theory]
    [InlineData("")]
    [InlineData("   ")]
    public void Resolve_BlankInput_ReturnsNull(string input)
    {
        var resolver = new MaterialMappingResolver(new InMemoryMaterialMappingRepository());
        Assert.Null(resolver.Resolve(input));
    }

    [Fact]
    public void Resolve_AfterRemove_ReturnsNull()
    {
        var repo = new InMemoryMaterialMappingRepository();
        repo.Set("Concreto", "V002");
        repo.Remove("Concreto");
        var resolver = new MaterialMappingResolver(repo);

        Assert.Null(resolver.Resolve("Concreto"));
    }

    [Fact]
    public void Resolve_AfterClear_ReturnsNull()
    {
        var repo = new InMemoryMaterialMappingRepository();
        repo.Set("Madera", "V003");
        repo.Clear();
        var resolver = new MaterialMappingResolver(repo);

        Assert.Null(resolver.Resolve("Madera"));
    }
}

// ── SubstringSuggester tests ──────────────────────────────────────────────────

public class SubstringSuggesterTests
{
    private static readonly Valor[] MaterialValues =
    [
        new Valor("V001", "Arcilla",  "COL_MAT"),
        new Valor("V002", "Concreto", "COL_MAT"),
        new Valor("V003", "Aluminio", "COL_MAT"),
        new Valor("V004", "Madera",   "COL_MAT"),
    ];

    [Fact]
    public void Suggest_NameContainsKeyword_ReturnsSuggestion()
    {
        var result = SubstringSuggester.Suggest("Bloque de arcilla", MaterialValues);
        Assert.Contains("V001", result);
    }

    [Fact]
    public void Suggest_KeywordContainsName_ReturnsSuggestion()
    {
        var result = SubstringSuggester.Suggest("Madera", MaterialValues);
        Assert.Contains("V004", result);
    }

    [Fact]
    public void Suggest_NoMatch_ReturnsEmpty()
    {
        var result = SubstringSuggester.Suggest("Polietileno expandido", MaterialValues);
        Assert.Empty(result);
    }

    [Theory]
    [InlineData("")]
    [InlineData("   ")]
    public void Suggest_BlankInput_ReturnsEmpty(string input)
    {
        var result = SubstringSuggester.Suggest(input, MaterialValues);
        Assert.Empty(result);
    }

    [Fact]
    public void Suggest_PartialWordMatch_ReturnsSuggestion()
    {
        // "aluminios" should match "Aluminio" via partial-word logic
        var result = SubstringSuggester.Suggest("panel aluminios", MaterialValues);
        Assert.Contains("V003", result);
    }
}

// ── InMemoryMaterialMappingRepository tests ───────────────────────────────────

public class InMemoryMaterialMappingRepositoryTests
{
    [Fact]
    public void Set_Then_GetAll_ContainsEntry()
    {
        var repo = new InMemoryMaterialMappingRepository();
        repo.Set("Mortero", "V010");

        Assert.True(repo.GetAll().ContainsKey("Mortero"));
        Assert.Equal("V010", repo.GetAll()["Mortero"]);
    }

    [Fact]
    public void Remove_ExistingEntry_IsGone()
    {
        var repo = new InMemoryMaterialMappingRepository();
        repo.Set("Mortero", "V010");
        repo.Remove("Mortero");

        Assert.False(repo.GetAll().ContainsKey("Mortero"));
    }

    [Fact]
    public void Clear_RemovesAllEntries()
    {
        var repo = new InMemoryMaterialMappingRepository();
        repo.Set("A", "1");
        repo.Set("B", "2");
        repo.Clear();

        Assert.Empty(repo.GetAll());
    }

    [Fact]
    public void Set_OverwritesExistingEntry()
    {
        var repo = new InMemoryMaterialMappingRepository();
        repo.Set("Concreto", "V001");
        repo.Set("Concreto", "V999");

        Assert.Equal("V999", repo.GetAll()["Concreto"]);
    }
}
