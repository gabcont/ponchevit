using System.Collections.Generic;
using System.Linq;
using Ponchevit.Domain.Model;

namespace Ponchevit.Domain.Graph;

/// <summary>
/// Utility to handle connections that do not contribute to the final code.
/// </summary>
public static class EmptyBridgeResolver
{
    /// <summary>
    /// Checks if a connection is an "empty bridge" (does not contribute code).
    /// </summary>
    public static bool IsEmptyBridge(Conexion conexion)
    {
        return string.IsNullOrEmpty(conexion.CodigoAportado);
    }

    /// <summary>
    /// Filters out empty bridges from a path of connections.
    /// Use this when only code-contributing nodes are needed.
    /// </summary>
    public static IEnumerable<Conexion> FilterBridges(IEnumerable<Conexion> path)
    {
        return path.Where(c => !IsEmptyBridge(c));
    }
}
