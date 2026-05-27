using Xunit;
using Ponchevit.Domain.Graph;
using Ponchevit.Domain.Model;
using System.Collections.Generic;
using System.Linq;

namespace Ponchevit.Tests.Domain.Graph;

public class EmptyBridgeResolverTests
{
    [Fact]
    public void IsEmptyBridge_ReturnsTrue_WhenCodigoAportadoIsEmptyOrNull()
    {
        var conn1 = new Conexion("C1", null, "", "COL1", null);
        var conn2 = new Conexion("C2", null, null!, "COL1", null);
        var conn3 = new Conexion("C3", null, "E4", "COL1", null);

        Assert.True(EmptyBridgeResolver.IsEmptyBridge(conn1));
        Assert.True(EmptyBridgeResolver.IsEmptyBridge(conn2));
        Assert.False(EmptyBridgeResolver.IsEmptyBridge(conn3));
    }

    [Fact]
    public void FilterBridges_RemovesEmptyBridges()
    {
        var path = new List<Conexion>
        {
            new Conexion("C1", null, "E4", "COL1", null),
            new Conexion("C2", "C1", "", "COL2", "V1"),
            new Conexion("C3", "C2", "1", "COL3", "V2")
        };

        var filtered = EmptyBridgeResolver.FilterBridges(path).ToList();

        Assert.Equal(2, filtered.Count);
        Assert.Equal("C1", filtered[0].IdConexion);
        Assert.Equal("C3", filtered[1].IdConexion);
    }
}
