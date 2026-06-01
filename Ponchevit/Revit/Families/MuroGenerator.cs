using System;
using Autodesk.Revit.DB;

namespace Ponchevit.Revit.Families;

/// <summary>
/// Generates a code-compliant Muro (wall) via Revit CompoundStructure.
/// Full implementation in Phase 4.7. Currently a registered stub so the
/// generator registry in Services.cs is populated from startup.
/// </summary>
public sealed class MuroGenerator : IFamilyGenerator
{
    public BuiltInCategory SupportedCategory => BuiltInCategory.OST_Walls;

    public void Generate(Document doc, GeneratorInput input)
    {
        throw new NotImplementedException(
            "MuroGenerator.Generate is implemented in Phase 4.7.");
    }
}
