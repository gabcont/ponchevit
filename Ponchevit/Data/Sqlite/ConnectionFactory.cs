using System;
using System.IO;
using System.Reflection;
using Microsoft.Data.Sqlite;

namespace Ponchevit.Data.Sqlite;

public class ConnectionFactory
{
    private readonly string _basePath;
    private const int ExpectedSchemaVersion = 1;

    public ConnectionFactory(string? basePath = null)
    {
        _basePath = basePath ?? Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location) 
                    ?? AppContext.BaseDirectory;
    }

    public SqliteConnection CreatePartidasConnection()
    {
        return CreateAndValidateConnection("partidas.db");
    }

    public SqliteConnection CreateCoveninConnection()
    {
        return CreateAndValidateConnection("covenin.db");
    }

    private SqliteConnection CreateAndValidateConnection(string dbName)
    {
        string dbPath = Path.Combine(_basePath, dbName);
        
        // For tests using :memory:, we don't check File.Exists
        if (!dbPath.Contains(":memory:") && !File.Exists(dbPath))
        {
            throw new FileNotFoundException($"Database file not found: {dbName} at {dbPath}");
        }

        var connection = new SqliteConnection($"Data Source={dbPath}");
        connection.Open();

        try
        {
            ValidateSchema(connection, dbName);
        }
        catch
        {
            connection.Close();
            throw;
        }

        return connection;
    }

    private void ValidateSchema(SqliteConnection connection, string dbName)
    {
        using var command = connection.CreateCommand();
        command.CommandText = "SELECT name FROM sqlite_master WHERE type='table' AND name='_meta'";
        var metaExists = command.ExecuteScalar();

        if (metaExists == null)
        {
            // _meta absent: pipeline predates versioning; skip version check.
            // Repository queries will surface schema mismatches naturally.
            return;
        }

        command.CommandText = "SELECT schema_version FROM _meta LIMIT 1";
        var version = command.ExecuteScalar();

        if (version == null || Convert.ToInt32(version) != ExpectedSchemaVersion)
        {
            throw new InvalidOperationException(
                $"Invalid schema version in {dbName}. Expected {ExpectedSchemaVersion}, found {version ?? "null"}.");
        }
    }
}
