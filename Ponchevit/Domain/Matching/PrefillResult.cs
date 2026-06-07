using System.Collections.Generic;

namespace Ponchevit.Domain.Matching;

public enum PrefillState
{
    AutoFilled,
    Ambiguous,
    Undetectable
}

public sealed record PrefillEntry(
    PrefillState State,
    string? SuggestedIdValor,
    string? SuggestedIdConexion = null
);

/// <summary>
/// Per-element recognition output: a map from IdColumna to the suggested fill state.
/// Produced by IElementRecognizer implementations; consumed by the Assign prefill flow.
/// </summary>
public sealed class PrefillResult
{
    public static readonly PrefillResult Empty = new PrefillResult(
        new Dictionary<string, PrefillEntry>());

    private readonly IReadOnlyDictionary<string, PrefillEntry> _entries;

    public PrefillResult(IReadOnlyDictionary<string, PrefillEntry> entries)
    {
        _entries = entries ?? throw new System.ArgumentNullException(nameof(entries));
    }

    public bool TryGet(string idColumna, out PrefillEntry entry)
        => _entries.TryGetValue(idColumna, out entry!);

    public IEnumerable<KeyValuePair<string, PrefillEntry>> Entries => _entries;
}
