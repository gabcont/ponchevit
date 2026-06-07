using System.Collections.Generic;
using Microsoft.Data.Sqlite;
using Ponchevit.Domain.Model;

namespace Ponchevit.Data.Sqlite;

public class SqliteCoveninRulesRepository : ICoveninRulesRepository
{
    private readonly ConnectionFactory _connectionFactory;
    private readonly IReadOnlyDictionary<string, Columna> _columnas;
    private readonly IReadOnlyDictionary<string, Valor> _valores;
    private readonly Dictionary<string, List<Conexion>> _conexionesCache = new();
    private List<Conexion>? _rootConexiones;
    private readonly Dictionary<string, Conexion?> _conexionesById = new();
    private readonly Dictionary<string, List<Conexion>> _conexionesByValorCache = new();

    public SqliteCoveninRulesRepository(ConnectionFactory connectionFactory)
    {
        _connectionFactory = connectionFactory;
        _columnas = LoadColumnas();
        _valores = LoadValores();
    }

    public IEnumerable<Columna> GetColumnas() => _columnas.Values;
    public IEnumerable<Valor> GetValores() => _valores.Values;

    public Columna? GetColumna(string idColumna) => _columnas.TryGetValue(idColumna, out var col) ? col : null;
    public Valor? GetValor(string idValor) => _valores.TryGetValue(idValor, out var val) ? val : null;

    public IEnumerable<Conexion> GetConexionesByParent(string? parentId)
    {
        if (string.IsNullOrEmpty(parentId))
        {
            return _rootConexiones ??= LoadConexiones(null);
        }

        if (_conexionesCache.TryGetValue(parentId, out var cached))
        {
            return cached;
        }

        var loaded = LoadConexiones(parentId);
        _conexionesCache[parentId] = loaded;
        return loaded;
    }

    public Conexion? GetConexionById(string idConexion)
    {
        if (string.IsNullOrEmpty(idConexion)) return null;

        if (_conexionesById.TryGetValue(idConexion, out var cached))
            return cached;

        using var connection = _connectionFactory.CreateCoveninConnection();
        using var command = connection.CreateCommand();
        command.CommandText =
            "SELECT Id_Conexion, Parent_Id, Codigo_Aportado, Id_Columna, Id_Valor_Asociado " +
            "FROM Covenin_Conexiones WHERE Id_Conexion = @id";
        command.Parameters.AddWithValue("@id", idConexion);

        using var reader = command.ExecuteReader();
        Conexion? result = null;
        if (reader.Read())
        {
            result = new Conexion(
                reader.GetString(0),
                reader.IsDBNull(1) ? null : reader.GetString(1),
                reader.GetString(2),
                reader.GetString(3),
                reader.IsDBNull(4) ? null : reader.GetString(4)
            );
        }

        _conexionesById[idConexion] = result;
        return result;
    }

    public IEnumerable<Conexion> GetConexionesByValorId(string idValorAsociado)
    {
        if (string.IsNullOrEmpty(idValorAsociado)) return Enumerable.Empty<Conexion>();

        if (_conexionesByValorCache.TryGetValue(idValorAsociado, out var cached))
            return cached;

        var list = new List<Conexion>();
        using var connection = _connectionFactory.CreateCoveninConnection();
        using var command = connection.CreateCommand();
        command.CommandText =
            "SELECT Id_Conexion, Parent_Id, Codigo_Aportado, Id_Columna, Id_Valor_Asociado " +
            "FROM Covenin_Conexiones WHERE Id_Valor_Asociado = @id";
        command.Parameters.AddWithValue("@id", idValorAsociado);

        using var reader = command.ExecuteReader();
        while (reader.Read())
        {
            list.Add(new Conexion(
                reader.GetString(0),
                reader.IsDBNull(1) ? null : reader.GetString(1),
                reader.GetString(2),
                reader.GetString(3),
                reader.IsDBNull(4) ? null : reader.GetString(4)
            ));
        }

        _conexionesByValorCache[idValorAsociado] = list;
        return list;
    }

    private Dictionary<string, Columna> LoadColumnas()
    {
        var dict = new Dictionary<string, Columna>();
        using var connection = _connectionFactory.CreateCoveninConnection();
        using var command = connection.CreateCommand();
        command.CommandText = "SELECT Id_Columna, Nombre FROM Covenin_Columnas";
        using var reader = command.ExecuteReader();
        while (reader.Read())
        {
            var id = reader.GetString(0);
            dict[id] = new Columna(id, reader.GetString(1));
        }
        return dict;
    }

    private Dictionary<string, Valor> LoadValores()
    {
        var dict = new Dictionary<string, Valor>();
        using var connection = _connectionFactory.CreateCoveninConnection();
        using var command = connection.CreateCommand();
        command.CommandText = "SELECT Id_Valor, Descripcion_Ui, Id_Columna, Num_Min, Num_Max, Unidad FROM Covenin_Valores";
        using var reader = command.ExecuteReader();
        while (reader.Read())
        {
            var id = reader.GetString(0);
            dict[id] = new Valor(
                id,
                reader.GetString(1),
                reader.GetString(2),
                reader.IsDBNull(3) ? null : reader.GetDouble(3),
                reader.IsDBNull(4) ? null : reader.GetDouble(4),
                reader.IsDBNull(5) ? null : reader.GetString(5)
            );
        }
        return dict;
    }

    private List<Conexion> LoadConexiones(string? parentId)
    {
        var list = new List<Conexion>();
        using var connection = _connectionFactory.CreateCoveninConnection();
        using var command = connection.CreateCommand();
        
        if (string.IsNullOrEmpty(parentId))
        {
            command.CommandText = "SELECT Id_Conexion, Parent_Id, Codigo_Aportado, Id_Columna, Id_Valor_Asociado FROM Covenin_Conexiones WHERE Parent_Id IS NULL OR Parent_Id = ''";
        }
        else
        {
            command.CommandText = "SELECT Id_Conexion, Parent_Id, Codigo_Aportado, Id_Columna, Id_Valor_Asociado FROM Covenin_Conexiones WHERE Parent_Id = @parentId";
            command.Parameters.AddWithValue("@parentId", parentId);
        }

        using var reader = command.ExecuteReader();
        while (reader.Read())
        {
            list.Add(new Conexion(
                reader.GetString(0),
                reader.IsDBNull(1) ? null : reader.GetString(1),
                reader.GetString(2),
                reader.GetString(3),
                reader.IsDBNull(4) ? null : reader.GetString(4)
            ));
        }
        return list;
    }
}
