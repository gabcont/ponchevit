using Ponchevit.Domain.Matching;

namespace Ponchevit.Ui.PartidaSelection;

/// <summary>
/// One line in the prefill report strip (Assign mode only).
/// Shows the column name, recognition state, and the detected value description when available.
/// </summary>
public sealed record PrefillReportLine(
    string ColumnaName,
    PrefillState State,
    string? DetectedValue
);
