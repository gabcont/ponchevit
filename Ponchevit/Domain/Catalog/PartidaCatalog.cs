namespace Ponchevit.Domain.Catalog;

using Ponchevit.Domain.Model;
using Ponchevit.Data;
using Ponchevit.Infrastructure;

public class PartidaCatalog
{
    private readonly List<Partida> _validPartidas = new();

    public PartidaCatalog(IPartidasRepository repository, ILog log)
    {
        var capitulos = repository.GetCapitulos().ToList();
        var subcapitulos = repository.GetSubcapitulos().ToList();
        var secciones = repository.GetSecciones().ToList();
        
        var resolver = new PartidaHierarchyResolver(capitulos, subcapitulos, secciones);

        foreach (var p in repository.GetPartidas())
        {
            if (IsAnomaly(p))
            {
                log.Warn($"Excluded anomaly partida (schema): {p.CodigoPartida}");
                continue;
            }

            var (cap, sub, sec) = resolver.Resolve(p.CodigoPartida);
            
            if (cap == null)
            {
                log.Warn($"Excluded anomaly partida (unresolved hierarchy): {p.CodigoPartida}");
                continue;
            }

            var resolvedPartida = p with 
            { 
                Capitulo = cap.Titulo,
                Subcapitulo = sub?.Titulo, 
                Seccion = sec?.Titulo 
            };
            
            _validPartidas.Add(resolvedPartida);
        }
    }

    public IReadOnlyList<Partida> GetPartidas() => _validPartidas;

    private bool IsAnomaly(Partida p)
    {
        if (string.IsNullOrWhiteSpace(p.CodigoPartida)) return true;
        if (p.CodigoPartida.Length != 10) return true;
        if (p.CodigoPartida.Contains('x', StringComparison.OrdinalIgnoreCase)) return true;
        return false;
    }
}
