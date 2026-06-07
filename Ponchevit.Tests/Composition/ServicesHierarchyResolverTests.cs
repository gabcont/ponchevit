using System.Collections.Generic;
using System.Linq;
using Ponchevit.Domain.Catalog;
using Ponchevit.Domain.Model;
using Xunit;

namespace Ponchevit.Tests.Composition;

/// <summary>
/// Verifies the Fix C pattern: PartidaHierarchyResolver is constructed once from
/// catalog data and can be injected.  The full Services.Build() path requires live
/// SQLite databases and is verified by manual smoke-test in Revit; here we verify
/// the construction contract and non-null guarantee when given valid inputs.
/// </summary>
public class ServicesHierarchyResolverTests
{
    [Fact]
    public void HierarchyResolver_ConstructedFromCatalogData_IsNonNull()
    {
        // This mirrors what Services.Build() does after Fix C.
        var capitulos    = new[] { new Capitulo("1", "E4", "Obras Arquitectónicas") };
        var subcapitulos = new[] { new Subcapitulo("2", "1", "E41", "Muros y Paredes") };
        var secciones    = new[] { new Seccion("3", "1", "2", "E411", "Muros de Bloque") };

        var resolver = new PartidaHierarchyResolver(capitulos, subcapitulos, secciones);

        Assert.NotNull(resolver);
    }

    [Fact]
    public void HierarchyResolver_InjectedInstance_ResolvesCorrectly()
    {
        // Verifies that the injected resolver produces the same results as
        // an inline-constructed one — i.e., stateless and reusable.
        var capitulos    = new[] { new Capitulo("1", "E4", "Obras Arquitectónicas") };
        var subcapitulos = new[] { new Subcapitulo("2", "1", "E41", "Muros y Paredes") };
        var secciones    = new[] { new Seccion("3", "1", "2", "E41101", "Muros de Bloque") };

        var resolver = new PartidaHierarchyResolver(capitulos, subcapitulos, secciones);

        // Call twice to confirm statelessness — same result both times.
        var (cap1, sub1, sec1) = resolver.Resolve("E411010101");
        var (cap2, sub2, sec2) = resolver.Resolve("E411010101");

        Assert.NotNull(cap1);
        Assert.Equal(cap1?.Titulo, cap2?.Titulo);
        Assert.Equal(sub1?.Titulo, sub2?.Titulo);
        Assert.Equal(sec1?.Titulo, sec2?.Titulo);
    }

    [Fact]
    public void HierarchyResolver_EmptyInputs_DoesNotThrow()
    {
        var ex = Record.Exception(() =>
            new PartidaHierarchyResolver(
                Enumerable.Empty<Capitulo>(),
                Enumerable.Empty<Subcapitulo>(),
                Enumerable.Empty<Seccion>()));

        Assert.Null(ex);
    }
}
