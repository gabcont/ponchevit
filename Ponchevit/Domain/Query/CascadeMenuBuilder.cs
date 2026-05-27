using System;
using System.Collections.Generic;
using System.Linq;
using Ponchevit.Domain.Model;

namespace Ponchevit.Domain.Query;

/// <summary>
/// Represents an option in a cascading menu.
/// </summary>
public record MenuOption(
    string IdConexion,
    string? IdValor,
    string Label,
    string CodigoAportado
);

/// <summary>
/// Represents a level in the cascading menu, corresponding to a DAG column.
/// </summary>
public record MenuLevel(
    Columna Columna,
    IReadOnlyList<MenuOption> Options
);

/// <summary>
/// Service that builds cascading menu levels from the COVENIN DAG.
/// </summary>
public class CascadeMenuBuilder
{
    private readonly ICoveninRulesProvider _rulesProvider;

    public CascadeMenuBuilder(ICoveninRulesProvider rulesProvider)
    {
        _rulesProvider = rulesProvider ?? throw new ArgumentNullException(nameof(rulesProvider));
    }

    /// <summary>
    /// Gets the next menu level based on the current selection.
    /// </summary>
    /// <param name="parentConnectionId">The ID of the currently selected connection, or null for the root level.</param>
    /// <returns>A MenuLevel object if children exist, otherwise null.</returns>
    public MenuLevel? GetNextLevel(string? parentConnectionId)
    {
        var children = _rulesProvider.GetChildren(parentConnectionId).ToList();
        if (!children.Any())
        {
            return null;
        }

        // All children at a given level must belong to the same column (logical step).
        var firstChild = children.First();
        var columna = _rulesProvider.GetColumna(firstChild.IdColumna);
        if (columna == null)
        {
            return null;
        }

        var options = new List<MenuOption>();
        foreach (var child in children)
        {
            string label = child.CodigoAportado;
            if (!string.IsNullOrEmpty(child.IdValorAsociado))
            {
                var valor = _rulesProvider.GetValor(child.IdValorAsociado);
                if (valor != null)
                {
                    label = valor.DescripcionUi;
                }
            }

            options.Add(new MenuOption(
                child.IdConexion,
                child.IdValorAsociado,
                label,
                child.CodigoAportado
            ));
        }

        return new MenuLevel(columna, options);
    }
}
