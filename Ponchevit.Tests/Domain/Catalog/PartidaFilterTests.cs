namespace Ponchevit.Tests.Domain.Catalog;

using Ponchevit.Domain.Catalog;
using Ponchevit.Domain.Model;
using Xunit;

public class PartidaFilterTests
{
    private readonly List<Partida> _partidas = new()
    {
        new Partida("E411011001", "m3", "P1", "Cap1"),
        new Partida("E411011002", "m3", "P2", "Cap1"),
        new Partida("E412011001", "m2", "P3", "Cap1"),
        new Partida("E511011001", "m", "P4", "Cap2")
    };

    [Fact]
    public void Filter_ByCapitulo_ReturnsMatching()
    {
        var filter = new PartidaFilter();
        var result = filter.Filter(_partidas, capituloCodigo: "E4").ToList();
        Assert.Equal(3, result.Count);
        Assert.All(result, p => Assert.StartsWith("E4", p.CodigoPartida));
    }

    [Fact]
    public void Filter_BySubcapitulo_ReturnsMatching()
    {
        var filter = new PartidaFilter();
        var result = filter.Filter(_partidas, subcapituloCodigo: "E412").ToList();
        Assert.Single(result);
        Assert.Equal("E412011001", result[0].CodigoPartida);
    }

    [Fact]
    public void Filter_BySeccion_ReturnsMatching()
    {
        var filter = new PartidaFilter();
        var result = filter.Filter(_partidas, seccionCodigo: "E41101").ToList();
        Assert.Equal(2, result.Count);
    }

    [Fact]
    public void Filter_Empty_ReturnsAll()
    {
        var filter = new PartidaFilter();
        var result = filter.Filter(_partidas).ToList();
        Assert.Equal(4, result.Count);
    }
}
