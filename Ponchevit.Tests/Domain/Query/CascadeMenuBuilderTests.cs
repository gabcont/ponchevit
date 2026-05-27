using System.Collections.Generic;
using System.Linq;
using Ponchevit.Domain.Model;
using Ponchevit.Domain.Query;
using Ponchevit.Data;
using Xunit;

namespace Ponchevit.Tests.Domain.Query;

public class CascadeMenuBuilderTests
{
    private class MockRulesRepository : ICoveninRulesRepository
    {
        public List<Conexion> Connections = new();
        public List<Columna> Columns = new();
        public List<Valor> Values = new();

        public IEnumerable<Conexion> GetConexionesByParent(string? parentId) =>
            Connections.Where(c => c.ParentId == parentId);

        public Columna? GetColumna(string idColumna) =>
            Columns.FirstOrDefault(c => c.IdColumna == idColumna);

        public Valor? GetValor(string idValor) =>
            Values.FirstOrDefault(v => v.IdValor == idValor);

        public IEnumerable<Columna> GetColumnas() => Columns;
        public IEnumerable<Valor> GetValores() => Values;
    }

    [Fact]
    public void GetNextLevel_ReturnsCorrectLevelAndOptions()
    {
        // Arrange
        var repository = new MockRulesRepository();
        repository.Columns.Add(new Columna("COL_1", "CAPITULO"));
        repository.Connections.Add(new Conexion("CON_1", null, "E4", "COL_1", null));
        repository.Connections.Add(new Conexion("CON_2", null, "E3", "COL_1", null));

        var builder = new CascadeMenuBuilder(repository);

        // Act
        var result = builder.GetNextLevel(null);

        // Assert
        Assert.NotNull(result);
        Assert.Equal("COL_1", result.Columna.IdColumna);
        Assert.Equal("CAPITULO", result.Columna.Nombre);
        Assert.Equal(2, result.Options.Count);
        Assert.Contains(result.Options, o => o.Label == "E4" && o.IdConexion == "CON_1");
        Assert.Contains(result.Options, o => o.Label == "E3" && o.IdConexion == "CON_2");
    }

    [Fact]
    public void GetNextLevel_UsesValueDescription_WhenAvailable()
    {
        // Arrange
        var repository = new MockRulesRepository();
        repository.Columns.Add(new Columna("COL_2", "SUB-CAPITULO"));
        repository.Values.Add(new Valor("VAL_1", "Albañilería", "COL_2"));
        repository.Connections.Add(new Conexion("CON_2", "CON_1", "1", "COL_2", "VAL_1"));

        var builder = new CascadeMenuBuilder(repository);

        // Act
        var result = builder.GetNextLevel("CON_1");

        // Assert
        Assert.NotNull(result);
        Assert.Equal("Albañilería", result.Options[0].Label);
        Assert.Equal("1", result.Options[0].CodigoAportado);
    }

    [Fact]
    public void GetNextLevel_ReturnsNull_WhenNoChildrenExist()
    {
        // Arrange
        var repository = new MockRulesRepository();
        var builder = new CascadeMenuBuilder(repository);

        // Act
        var result = builder.GetNextLevel("CON_UNKNOWN");

        // Assert
        Assert.Null(result);
    }
}
