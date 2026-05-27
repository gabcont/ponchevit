using System.Collections.Generic;
using System.Linq;
using Ponchevit.Domain.Model;
using Ponchevit.Domain.Graph;

namespace Ponchevit.Domain.Query;

/// <summary>
/// Result containing the COVENIN prefixes for each hierarchical level.
/// </summary>
public record PrefixPathResult(
    CodigoCovenin Capitulo,
    CodigoCovenin Subcapitulo,
    CodigoCovenin Seccion
);

/// <summary>
/// Service that derives Capítulo/Subcapítulo/Sección prefixes from a DAG path.
/// </summary>
public class PrefixPathQuery
{
    private readonly CodeAssembler _assembler = new();

    /// <summary>
    /// Derives the prefixes for Capítulo, Subcapítulo, and Sección from the given connection path.
    /// Assumes the first three logical steps in the path correspond to these levels.
    /// </summary>
    public PrefixPathResult GetPrefixes(IEnumerable<Conexion> path)
    {
        var connections = path.ToList();
        
        // Assemble the code at the first three steps of the path
        var capitulo = _assembler.Assemble(connections.Take(1));
        var subcapitulo = _assembler.Assemble(connections.Take(2));
        var seccion = _assembler.Assemble(connections.Take(3));

        return new PrefixPathResult(capitulo, subcapitulo, seccion);
    }
}
