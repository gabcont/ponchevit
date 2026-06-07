using System.Collections.Generic;

namespace Ponchevit.Revit;

/// <summary>
/// Returns the list of material names present in the active Revit project.
/// Interface lives in Revit/ because its only purpose is to abstract a
/// FilteredElementCollector call; both MapeoMaterialesCommand and
/// AgregarFamiliaCommand consume the same service.
///
/// Implementation: <see cref="ProjectMaterialQuery"/>.
/// See ADR 2026-06-06 — Project-material query service (option B).
/// Fix E: no Document parameter — implementation obtains Document via injected IRevitContext.
/// </summary>
public interface IProjectMaterialQuery
{
    /// <summary>
    /// Returns an alphabetically-sorted, de-duped list of non-empty material names
    /// from the active document. Returns an empty list when no document is active.
    /// </summary>
    IReadOnlyList<string> GetProjectMaterials();
}
