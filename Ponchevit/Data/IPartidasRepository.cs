namespace Ponchevit.Data;

using Ponchevit.Domain.Model;

public interface IPartidasRepository
{
    IEnumerable<Capitulo> GetCapitulos();
    IEnumerable<Subcapitulo> GetSubcapitulos();
    IEnumerable<Seccion> GetSecciones();
    IEnumerable<Partida> GetPartidas();
}
