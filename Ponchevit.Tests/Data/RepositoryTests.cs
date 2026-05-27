using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.Data.Sqlite;
using Ponchevit.Data.Sqlite;
using Ponchevit.Domain.Model;
using Xunit;

namespace Ponchevit.Tests.Data;

public class RepositoryTests : IDisposable
{
    private readonly string _tempPath;
    private const string MetaSchema = "CREATE TABLE _meta (schema_version INTEGER); INSERT INTO _meta (schema_version) VALUES (1);";

    public RepositoryTests()
    {
        _tempPath = Path.Combine(Path.GetTempPath(), "PonchevitTests_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_tempPath);
    }

    public void Dispose()
    {
        SqliteConnection.ClearAllPools();
        if (Directory.Exists(_tempPath))
        {
            Directory.Delete(_tempPath, true);
        }
    }

    private void CreateDb(string name, string sql)
    {
        string path = Path.Combine(_tempPath, name);
        using var connection = new SqliteConnection($"Data Source={path}");
        connection.Open();
        using var cmd = connection.CreateCommand();
        cmd.CommandText = MetaSchema + sql;
        cmd.ExecuteNonQuery();
    }

    [Fact]
    public void SqlitePartidasRepository_LoadsDataCorrectly()
    {
        CreateDb("partidas.db", @"
            CREATE TABLE Capitulos (id TEXT, codigo TEXT, titulo TEXT);
            CREATE TABLE Subcapitulos (id TEXT, capitulo_id TEXT, codigo TEXT, titulo TEXT);
            CREATE TABLE Secciones (id TEXT, capitulo_id TEXT, subcapitulo_id TEXT, codigo TEXT, titulo TEXT);
            CREATE TABLE Partidas (codigo_partida TEXT, unidad TEXT, descripcion TEXT, capitulo TEXT);
            
            INSERT INTO Capitulos VALUES ('c1', 'E4', 'Obras Arquitectónicas');
            INSERT INTO Subcapitulos VALUES ('sc1', 'c1', 'E41', 'Albañilería');
            INSERT INTO Secciones VALUES ('s1', 'c1', 'sc1', 'E411', 'Muros y Tabiques');
            INSERT INTO Partidas VALUES ('E411011000', 'm2', 'Muro de ladrillo', 'Obras Arquitectónicas');
        ");

        var factory = new ConnectionFactory(_tempPath);
        var repo = new SqlitePartidasRepository(factory);

        Assert.Single(repo.GetCapitulos());
        Assert.Equal("E4", repo.GetCapitulos().First().Codigo);
        Assert.Single(repo.GetSubcapitulos());
        Assert.Single(repo.GetSecciones());
        Assert.Single(repo.GetPartidas());
        Assert.Equal("E411011000", repo.GetPartidas().First().CodigoPartida);
    }

    [Fact]
    public void SqliteCoveninRulesRepository_LoadsAndCachesCorrectly()
    {
        CreateDb("covenin.db", @"
            CREATE TABLE Covenin_Columnas (Id_Columna TEXT, Nombre TEXT);
            CREATE TABLE Covenin_Valores (Id_Valor TEXT, Descripcion_Ui TEXT, Id_Columna TEXT, Num_Min REAL, Num_Max REAL, Unidad TEXT);
            CREATE TABLE Covenin_Conexiones (Id_Conexion TEXT, Parent_Id TEXT, Codigo_Aportado TEXT, Id_Columna TEXT, Id_Valor_Asociado TEXT);
            
            INSERT INTO Covenin_Columnas VALUES ('col1', 'Material');
            INSERT INTO Covenin_Valores VALUES ('val1', 'Ladrillo', 'col1', NULL, NULL, NULL);
            INSERT INTO Covenin_Conexiones VALUES ('conn1', NULL, '1', 'col1', 'val1');
            INSERT INTO Covenin_Conexiones VALUES ('conn2', 'conn1', '2', 'col1', 'val1');
        ");

        var factory = new ConnectionFactory(_tempPath);
        var repo = new SqliteCoveninRulesRepository(factory);

        Assert.Single(repo.GetColumnas());
        Assert.Single(repo.GetValores());
        
        var roots = repo.GetConexionesByParent(null).ToList();
        Assert.Single(roots);
        Assert.Equal("conn1", roots[0].IdConexion);

        var children = repo.GetConexionesByParent("conn1");
        Assert.Single(children);
        Assert.Equal("conn2", children.First().IdConexion);
        
        // Test caching: call again and it should come from cache (internally)
        var childrenCached = repo.GetConexionesByParent("conn1");
        Assert.Same(children, childrenCached);
    }

    [Fact]
    public void ConnectionFactory_ThrowsOnInvalidVersion()
    {
        string dbPath = Path.Combine(_tempPath, "partidas.db");
        using (var connection = new SqliteConnection($"Data Source={dbPath}"))
        {
            connection.Open();
            using var cmd = connection.CreateCommand();
            cmd.CommandText = "CREATE TABLE _meta (schema_version INTEGER); INSERT INTO _meta (schema_version) VALUES (2);";
            cmd.ExecuteNonQuery();
        }

        var factory = new ConnectionFactory(_tempPath);
        Assert.Throws<InvalidOperationException>(() => factory.CreatePartidasConnection());
    }

    [Fact]
    public void ConnectionFactory_ThrowsOnMissingMeta()
    {
        string dbPath = Path.Combine(_tempPath, "partidas.db");
        using (var connection = new SqliteConnection($"Data Source={dbPath}"))
        {
            connection.Open();
            using var cmd = connection.CreateCommand();
            cmd.CommandText = "CREATE TABLE Dummy (id INTEGER);";
            cmd.ExecuteNonQuery();
        }

        var factory = new ConnectionFactory(_tempPath);
        Assert.Throws<InvalidOperationException>(() => factory.CreatePartidasConnection());
    }
}
