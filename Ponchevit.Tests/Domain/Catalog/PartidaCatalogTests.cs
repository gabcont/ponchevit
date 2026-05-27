namespace Ponchevit.Tests.Domain.Catalog;

using Ponchevit.Domain.Catalog;
using Ponchevit.Domain.Model;
using Ponchevit.Data;
using Ponchevit.Infrastructure;
using Xunit;

public class PartidaCatalogTests
{
    private class MockRepo : IPartidasRepository
    {
        public List<Capitulo> Capitulos { get; set; } = new();
        public List<Subcapitulo> Subcapitulos { get; set; } = new();
        public List<Seccion> Secciones { get; set; } = new();
        public List<Partida> Partidas { get; set; } = new();

        public IEnumerable<Capitulo> GetCapitulos() => Capitulos;
        public IEnumerable<Subcapitulo> GetSubcapitulos() => Subcapitulos;
        public IEnumerable<Seccion> GetSecciones() => Secciones;
        public IEnumerable<Partida> GetPartidas() => Partidas;
    }

    private class MockLog : ILog
    {
        public List<string> Warnings { get; } = new();
        public void Info(string message) { }
        public void Warn(string message) => Warnings.Add(message);
        public void Error(string message, Exception? ex = null) { }
    }

    [Fact]
    public void Load_AttachesHierarchyAndFiltersAnomalies()
    {
        // Arrange
        var repo = new MockRepo();
        repo.Capitulos.Add(new Capitulo("1", "E4", "EDIFICACIONES"));
        repo.Subcapitulos.Add(new Subcapitulo("2", "1", "E411", "ESTRUCTURAS"));
        repo.Secciones.Add(new Seccion("3", "1", "2", "E41101", "CONCRETO ARMADO"));

        repo.Partidas.Add(new Partida("E411011015", "m3", "CONCRETO", "Temp")); // Valid
        repo.Partidas.Add(new Partida("E41101", "m3", "Short", "Temp")); // Anomaly (length != 10)
        repo.Partidas.Add(new Partida("E015xxx5xx", "m3", "Placeholder", "Temp")); // Anomaly (xxx)
        repo.Partidas.Add(new Partida("X999999999", "m3", "NoCap", "Temp")); // Anomaly (unresolved)

        var log = new MockLog();

        // Act
        var catalog = new PartidaCatalog(repo, log);
        var partidas = catalog.GetPartidas();

        // Assert
        Assert.Single(partidas);
        var p = partidas[0];
        Assert.Equal("E411011015", p.CodigoPartida);
        Assert.Equal("EDIFICACIONES", p.Capitulo);
        Assert.Equal("ESTRUCTURAS", p.Subcapitulo);
        Assert.Equal("CONCRETO ARMADO", p.Seccion);
        
        Assert.Equal(3, log.Warnings.Count);
    }
}
