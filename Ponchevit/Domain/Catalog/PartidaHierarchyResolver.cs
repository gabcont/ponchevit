namespace Ponchevit.Domain.Catalog;

using Ponchevit.Domain.Model;

public class PartidaHierarchyResolver
{
    private readonly List<Capitulo> _capitulos;
    private readonly List<Subcapitulo> _subcapitulos;
    private readonly List<Seccion> _secciones;

    public PartidaHierarchyResolver(
        IEnumerable<Capitulo> capitulos,
        IEnumerable<Subcapitulo> subcapitulos,
        IEnumerable<Seccion> secciones)
    {
        _capitulos = capitulos.OrderByDescending(c => c.Codigo.Length).ToList();
        _subcapitulos = subcapitulos.OrderByDescending(s => s.Codigo.Length).ToList();
        _secciones = secciones.OrderByDescending(s => s.Codigo.Length).ToList();
    }

    public (Capitulo? Capitulo, Subcapitulo? Subcapitulo, Seccion? Seccion) Resolve(string codigoPartida)
    {
        var seccion = _secciones.FirstOrDefault(s => codigoPartida.StartsWith(s.Codigo));
        var subcapitulo = _subcapitulos.FirstOrDefault(s => codigoPartida.StartsWith(s.Codigo));
        var capitulo = _capitulos.FirstOrDefault(c => codigoPartida.StartsWith(c.Codigo));

        return (capitulo, subcapitulo, seccion);
    }
}
