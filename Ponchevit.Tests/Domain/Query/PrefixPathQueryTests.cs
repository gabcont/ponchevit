using System.Collections.Generic;
using Ponchevit.Domain.Model;
using Ponchevit.Domain.Query;
using Xunit;

namespace Ponchevit.Tests.Domain.Query;

public class PrefixPathQueryTests
{
    [Fact]
    public void GetPrefixes_ReturnsCorrectPrefixes()
    {
        // Arrange
        var query = new PrefixPathQuery();
        var path = new List<Conexion>
        {
            new("CON_1", null, "E4", "COL_1", null),
            new("CON_2", "CON_1", "1", "COL_2", "VAL_1"),
            new("CON_3", "CON_2", "1", "COL_3", "VAL_2"),
            new("CON_4", "CON_3", "01", "COL_4", "VAL_3")
        };

        // Act
        var result = query.GetPrefixes(path);

        // Assert
        Assert.Equal("E4", result.Capitulo.Value);
        Assert.Equal("E41", result.Subcapitulo.Value);
        Assert.Equal("E411", result.Seccion.Value);
    }

    [Fact]
    public void GetPrefixes_WithEmptyBridges_ReturnsCorrectPrefixes()
    {
        // Arrange
        var query = new PrefixPathQuery();
        var path = new List<Conexion>
        {
            new("CON_1", null, "E4", "COL_1", null),
            new("CON_2", "CON_1", "", "COL_2", "VAL_1"), // Empty bridge
            new("CON_3", "CON_2", "1", "COL_3", "VAL_2"),
            new("CON_4", "CON_3", "1", "COL_4", "VAL_3")
        };

        // Act
        var result = query.GetPrefixes(path);

        // Assert
        Assert.Equal("E4", result.Capitulo.Value);
        Assert.Equal("E4", result.Subcapitulo.Value); // E4 + ""
        Assert.Equal("E41", result.Seccion.Value);   // E4 + "" + 1
    }

    [Fact]
    public void GetPrefixes_HandlesShortPath()
    {
        // Arrange
        var query = new PrefixPathQuery();
        var path = new List<Conexion>
        {
            new("CON_1", null, "E4", "COL_1", null)
        };

        // Act
        var result = query.GetPrefixes(path);

        // Assert
        Assert.Equal("E4", result.Capitulo.Value);
        Assert.Equal("E4", result.Subcapitulo.Value);
        Assert.Equal("E4", result.Seccion.Value);
    }
}
