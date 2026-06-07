using Ponchevit.Domain.Model;

namespace Ponchevit.Data;

public interface ICoveninRulesRepository
{
    IEnumerable<Columna> GetColumnas();
    IEnumerable<Valor> GetValores();
    IEnumerable<Conexion> GetConexionesByParent(string? parentId);
    Columna? GetColumna(string idColumna);
    Valor? GetValor(string idValor);
    Conexion? GetConexionById(string idConexion);
    IEnumerable<Conexion> GetConexionesByValorId(string idValorAsociado);
}
