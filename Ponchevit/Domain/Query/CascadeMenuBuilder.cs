using System;
using System.Collections.Generic;
using System.Linq;
using Ponchevit.Domain.Model;
using Ponchevit.Data;

namespace Ponchevit.Domain.Query;

/// <summary>
/// Represents an option in a cascading menu.
/// </summary>
public record MenuOption(
    string IdConexion,
    string? IdValor,
    string Label,
    string CodigoAportado,
    double? NumMin = null,
    double? NumMax = null,
    Ponchevit.Domain.Model.Valor? ValorData = null
);

/// <summary>
/// Represents a level in the cascading menu, corresponding to a DAG column.
/// </summary>
public record MenuLevel(
    Columna Columna,
    IReadOnlyList<MenuOption> Options,
    bool IsNumericColumn = false
);

/// <summary>
/// Service that builds cascading menu levels from the COVENIN DAG.
/// </summary>
public class CascadeMenuBuilder
{
    private readonly ICoveninRulesRepository _rulesRepository;

    public CascadeMenuBuilder(ICoveninRulesRepository rulesRepository)
    {
        _rulesRepository = rulesRepository ?? throw new ArgumentNullException(nameof(rulesRepository));
    }

    /// <summary>
    /// Gets the next menu level based on the current selection.
    /// </summary>
    /// <param name="parentConnectionId">The ID of the currently selected connection, or null for the root level.</param>
    /// <returns>A MenuLevel object if children exist, otherwise null.</returns>
    public MenuLevel? GetNextLevel(string? parentConnectionId)
    {
        var children = _rulesRepository.GetConexionesByParent(parentConnectionId).ToList();
        if (!children.Any())
        {
            return null;
        }

        // All children at a given level must belong to the same column (logical step).
        var firstChild = children.First();
        var columna = _rulesRepository.GetColumna(firstChild.IdColumna);
        if (columna == null)
        {
            return null;
        }

        var options = new List<MenuOption>();
        foreach (var child in children)
        {
            string label = child.CodigoAportado;
            Valor? valorData = null;
            if (!string.IsNullOrEmpty(child.IdValorAsociado))
            {
                valorData = _rulesRepository.GetValor(child.IdValorAsociado);
                if (valorData != null)
                {
                    label = valorData.DescripcionUi;
                }
            }

            options.Add(new MenuOption(
                child.IdConexion,
                child.IdValorAsociado,
                label,
                child.CodigoAportado,
                NumMin: valorData?.NumMin,
                NumMax: valorData?.NumMax,
                ValorData: valorData
            ));
        }

        bool isNumericColumn = options.Count > 0
            && options.All(o => o.NumMin.HasValue && o.NumMax.HasValue);

        return new MenuLevel(columna, options, isNumericColumn);
    }
}
