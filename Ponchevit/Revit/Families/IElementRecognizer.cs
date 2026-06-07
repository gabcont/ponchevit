using Autodesk.Revit.DB;
using Ponchevit.Domain.Matching;

namespace Ponchevit.Revit.Families;

/// <summary>
/// Strategy interface for recognising an existing Revit element's topology and
/// suggesting COVENIN DAG parameter values for the Assign prefill flow.
/// One implementation per supported element category (MuroRecognizer for MVP).
///
/// Mirrors IFamilyGenerator: same BuiltInCategory + CanRecognize predicate pattern.
/// To add support for a new element type, implement this interface and register the
/// instance in the element-module registry in Composition/Services.cs.
/// </summary>
public interface IElementRecognizer
{
    BuiltInCategory SupportedCategory { get; }

    /// <summary>
    /// Returns true when this recognizer can attempt recognition for the given topology.
    /// Used to route the topology to the correct recognizer without needing to know the
    /// BuiltInCategory directly (topology.Category is a string, not an enum).
    /// </summary>
    bool CanRecognize(ElementTopology topology);

    /// <summary>
    /// Analyses the topology and returns a per-IdColumna map of suggested values.
    /// Never throws — returns PrefillResult.Empty on any unrecognizable input.
    /// </summary>
    PrefillResult Recognize(ElementTopology topology);
}
