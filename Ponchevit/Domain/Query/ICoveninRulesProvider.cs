using System.Collections.Generic;
using Ponchevit.Domain.Model;

namespace Ponchevit.Domain.Query;

/// <summary>
/// Abstraction for fetching COVENIN rules and DAG connections.
/// </summary>
public interface ICoveninRulesProvider
{
    /// <summary>
    /// Gets all child connections for a given parent connection ID.
    /// If parentId is null, returns root connections (Capítulos).
    /// </summary>
    IEnumerable<Conexion> GetChildren(string? parentId);

    /// <summary>
    /// Gets column metadata by ID.
    /// </summary>
    Columna? GetColumna(string idColumna);

    /// <summary>
    /// Gets value metadata by ID.
    /// </summary>
    Valor? GetValor(string idValor);
}
