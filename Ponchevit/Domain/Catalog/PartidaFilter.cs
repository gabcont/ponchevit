namespace Ponchevit.Domain.Catalog;

using Ponchevit.Domain.Model;

public class PartidaFilter
{
    public IEnumerable<Partida> Filter(
        IEnumerable<Partida> partidas,
        string? capituloCodigo = null,
        string? subcapituloCodigo = null,
        string? seccionCodigo = null)
    {
        // In COVENIN E4, codes are hierarchical: Seccion starts with Subcapitulo, which starts with Capitulo.
        // We filter by the most specific prefix provided.
        
        string? effectivePrefix = seccionCodigo ?? subcapituloCodigo ?? capituloCodigo;

        if (string.IsNullOrWhiteSpace(effectivePrefix))
        {
            return partidas;
        }

        return partidas.Where(p => p.CodigoPartida.StartsWith(effectivePrefix));
    }
}
