using Ponchevit.Domain.Model;

namespace Ponchevit.Ui.AgregarFamilia;

/// <summary>
/// Thin UI wrapper around <see cref="Partida"/> that adds constructibility metadata
/// for the right panel's display (greyed-out style, selectability).
/// </summary>
public sealed class PartidaDisplayItem
{
    public Partida Partida { get; }

    /// <summary>
    /// True when a complete DAG path exists whose assembled code equals
    /// <see cref="Partida.CodigoPartida"/>.  False for partidas whose capítulo
    /// has no COVENIN rules, or those that exist in the flat catalog but cannot be
    /// reached via any valid DAG traversal (known typos in the source list).
    /// Distinct from <em>IsGenerable</em> (a generator module exists for this category).
    /// </summary>
    public bool CanBeConstructed { get; }

    // Convenience pass-throughs for XAML binding.
    public string CodigoPartida => Partida.CodigoPartida;
    public string Descripcion   => Partida.Descripcion;
    public string Unidad        => Partida.Unidad;

    public PartidaDisplayItem(Partida partida, bool canBeConstructed)
    {
        Partida          = partida ?? throw new System.ArgumentNullException(nameof(partida));
        CanBeConstructed = canBeConstructed;
    }
}
