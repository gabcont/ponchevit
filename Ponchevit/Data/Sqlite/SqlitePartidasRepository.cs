using System.Collections.Generic;
using Microsoft.Data.Sqlite;
using Ponchevit.Domain.Model;

namespace Ponchevit.Data.Sqlite;

public class SqlitePartidasRepository : IPartidasRepository
{
    private readonly IReadOnlyList<Capitulo> _capitulos;
    private readonly IReadOnlyList<Subcapitulo> _subcapitulos;
    private readonly IReadOnlyList<Seccion> _secciones;
    private readonly IReadOnlyList<Partida> _partidas;

    public SqlitePartidasRepository(ConnectionFactory connectionFactory)
    {
        _capitulos = LoadCapitulos(connectionFactory);
        _subcapitulos = LoadSubcapitulos(connectionFactory);
        _secciones = LoadSecciones(connectionFactory);
        _partidas = LoadPartidas(connectionFactory);
    }

    public IEnumerable<Capitulo> GetCapitulos() => _capitulos;
    public IEnumerable<Subcapitulo> GetSubcapitulos() => _subcapitulos;
    public IEnumerable<Seccion> GetSecciones() => _secciones;
    public IEnumerable<Partida> GetPartidas() => _partidas;

    private static IReadOnlyList<Capitulo> LoadCapitulos(ConnectionFactory factory)
    {
        var list = new List<Capitulo>();
        using var connection = factory.CreatePartidasConnection();
        using var command = connection.CreateCommand();
        command.CommandText = "SELECT id, codigo, titulo FROM Capitulos";
        using var reader = command.ExecuteReader();
        while (reader.Read())
        {
            list.Add(new Capitulo(
                reader.GetString(0),
                reader.GetString(1),
                reader.GetString(2)
            ));
        }
        return list;
    }

    private static IReadOnlyList<Subcapitulo> LoadSubcapitulos(ConnectionFactory factory)
    {
        var list = new List<Subcapitulo>();
        using var connection = factory.CreatePartidasConnection();
        using var command = connection.CreateCommand();
        command.CommandText = "SELECT id, capitulo_id, codigo, titulo FROM Subcapitulos";
        using var reader = command.ExecuteReader();
        while (reader.Read())
        {
            list.Add(new Subcapitulo(
                reader.GetString(0),
                reader.GetString(1),
                reader.GetString(2),
                reader.GetString(3)
            ));
        }
        return list;
    }

    private static IReadOnlyList<Seccion> LoadSecciones(ConnectionFactory factory)
    {
        var list = new List<Seccion>();
        using var connection = factory.CreatePartidasConnection();
        using var command = connection.CreateCommand();
        command.CommandText = "SELECT id, capitulo_id, subcapitulo_id, codigo, titulo FROM Secciones";
        using var reader = command.ExecuteReader();
        while (reader.Read())
        {
            list.Add(new Seccion(
                reader.GetString(0),
                reader.GetString(1),
                reader.GetString(2),
                reader.GetString(3),
                reader.GetString(4)
            ));
        }
        return list;
    }

    private static IReadOnlyList<Partida> LoadPartidas(ConnectionFactory factory)
    {
        var list = new List<Partida>();
        using var connection = factory.CreatePartidasConnection();
        using var command = connection.CreateCommand();
        command.CommandText = "SELECT codigo_partida, unidad, descripcion, capitulo FROM Partidas";
        using var reader = command.ExecuteReader();
        while (reader.Read())
        {
            list.Add(new Partida(
                reader.GetString(0),
                reader.GetString(1),
                reader.GetString(2),
                reader.GetString(3)
            ));
        }
        return list;
    }
}
