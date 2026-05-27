using System;
using System.Collections.Generic;
using System.Text;
using Ponchevit.Domain.Model;

namespace Ponchevit.Domain.Graph;

/// <summary>
/// Service that assembles COVENIN codes from a path of DAG connections.
/// </summary>
public class CodeAssembler
{
    public const int MaxCodeLength = 10;

    /// <summary>
    /// Concatenates the CodigoAportado from each connection in the path,
    /// enforcing a 10-digit maximum.
    /// </summary>
    public CodigoCovenin Assemble(IEnumerable<Conexion> path)
    {
        if (path == null) return new CodigoCovenin(string.Empty);

        var sb = new StringBuilder();
        foreach (var conexion in path)
        {
            string contribution = conexion.CodigoAportado;
            if (string.IsNullOrEmpty(contribution)) continue;

            int remaining = MaxCodeLength - sb.Length;
            if (remaining <= 0) break;

            if (contribution.Length > remaining)
            {
                sb.Append(contribution.Substring(0, remaining));
                break; // Reached cap
            }
            
            sb.Append(contribution);
        }

        return new CodigoCovenin(sb.ToString());
    }

    /// <summary>
    /// Computes the prefix code up to the specified connection ID within the given path.
    /// </summary>
    public CodigoCovenin ComputePrefix(IEnumerable<Conexion> path, string targetConnectionId)
    {
        if (path == null) return new CodigoCovenin(string.Empty);

        var partialPath = new List<Conexion>();
        bool found = false;
        foreach (var conexion in path)
        {
            partialPath.Add(conexion);
            if (conexion.IdConexion == targetConnectionId)
            {
                found = true;
                break;
            }
        }
        return found ? Assemble(partialPath) : new CodigoCovenin(string.Empty);
    }
}
