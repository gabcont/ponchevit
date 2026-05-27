namespace Ponchevit.Tests.Domain.Catalog;

using Ponchevit.Domain.Catalog;
using Ponchevit.Domain.Model;
using Xunit;

public class PartidaHierarchyResolverTests
{
    [Fact]
    public void Resolve_ShouldReturnLongestMatchingPrefix()
    {
        // Arrange
        var capitulos = new[] { new Capitulo("1", "E4", "Cap 1") };
        var subcapitulos = new[] 
        { 
            new Subcapitulo("2", "1", "E411", "Sub 1"),
            new Subcapitulo("3", "1", "E412", "Sub 2")
        };
        var secciones = new[] 
        { 
            new Seccion("4", "1", "2", "E41101", "Sec 1"),
            new Seccion("5", "1", "2", "E4110110", "Sec 1.1") // More specific
        };

        var resolver = new PartidaHierarchyResolver(capitulos, subcapitulos, secciones);

        // Act
        var (cap, sub, sec) = resolver.Resolve("E411011015");

        // Assert
        Assert.NotNull(cap);
        Assert.Equal("E4", cap.Codigo);
        Assert.NotNull(sub);
        Assert.Equal("E411", sub.Codigo);
        Assert.NotNull(sec);
        Assert.Equal("E4110110", sec.Codigo);
    }

    [Fact]
    public void Resolve_ShouldReturnNullWhenNoMatch()
    {
        // Arrange
        var resolver = new PartidaHierarchyResolver(
            Enumerable.Empty<Capitulo>(),
            Enumerable.Empty<Subcapitulo>(),
            Enumerable.Empty<Seccion>());

        // Act
        var (cap, sub, sec) = resolver.Resolve("X999");

        // Assert
        Assert.Null(cap);
        Assert.Null(sub);
        Assert.Null(sec);
    }
}
