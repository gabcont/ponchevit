using System;
using System.IO;
using System.Reflection;
using System.Text;
using Autodesk.Revit.DB;

namespace Ponchevit.Revit.SharedParameters;

/// <summary>
/// Single source of truth for the 4 COVENIN shared-parameter GUIDs and their names.
/// These GUIDs are PERMANENT — changing them orphans every existing .rvt that uses
/// Ponchevit (old values become invisible to the plugin while remaining as orphan params).
/// SharedParameters.txt is regenerated from these constants; never edit the file manually.
/// See ADR 2026-05-31 — GUID source-of-truth.
/// </summary>
public static class CoveninParameters
{
    // ── Stable GUIDs — NEVER CHANGE ────────────────────────────────────────────
    public static readonly Guid CapituloGuid       = new("A1B2C3D4-E5F6-7890-ABCD-EF1234567890");
    public static readonly Guid SubcapituloGuid    = new("B2C3D4E5-F6A7-8901-BCDE-F01234567891");
    public static readonly Guid SeccionGuid        = new("C3D4E5F6-A7B8-9012-CDEF-012345678912");
    public static readonly Guid CodigoCompletoGuid = new("D4E5F6A7-B8C9-0123-DEFA-123456789012");
    // ───────────────────────────────────────────────────────────────────────────

    public const string CapituloName       = "Capitulo_COVENIN";
    public const string SubcapituloName    = "Subcapitulo_COVENIN";
    public const string SeccionName        = "Seccion_COVENIN";
    public const string CodigoCompletoName = "Codigo_COVENIN_Completo";

    private const string GroupName = "Ponchevit COVENIN";

    private static readonly (Guid Guid, string Name)[] AllParams =
    [
        (CapituloGuid,       CapituloName),
        (SubcapituloGuid,    SubcapituloName),
        (SeccionGuid,        SeccionName),
        (CodigoCompletoGuid, CodigoCompletoName),
    ];

    /// <summary>
    /// Idempotent: binds the 4 shared parameters to all model categories in the given
    /// document. Safe to call on every command execution. Regenerates SharedParameters.txt
    /// beside the DLL if the file is missing.
    /// Must be called inside or before a transaction — creates its own transaction internally.
    /// </summary>
    public static void EnsureBoundToProject(Document doc)
    {
        var app = doc.Application;
        string filePath = GetSharedParametersFilePath();

        if (!File.Exists(filePath))
            GenerateSharedParametersFile(filePath);

        string previousFile = app.SharedParametersFilename;
        app.SharedParametersFilename = filePath;

        try
        {
            DefinitionFile? defFile = app.OpenSharedParameterFile();
            if (defFile == null)
                throw new InvalidOperationException(
                    $"Could not open shared parameter file at: {filePath}");

            DefinitionGroup group = defFile.Groups.get_Item(GroupName)
                                    ?? defFile.Groups.Create(GroupName);

            CategorySet categories = BuildModelCategorySet(doc);

            using var t = new Transaction(doc, "Bind COVENIN shared parameters");
            t.Start();

            foreach (var (guid, name) in AllParams)
            {
                ExternalDefinition? extDef =
                    group.Definitions.get_Item(name) as ExternalDefinition;

                if (extDef == null)
                {
                    var options = new ExternalDefinitionCreationOptions(name, SpecTypeId.String.Text)
                    {
                        GUID = guid,
                        UserModifiable = true,
                        Visible = true,
                    };
                    extDef = group.Definitions.Create(options) as ExternalDefinition;
                }

                if (extDef == null)
                    continue;

                if (!doc.ParameterBindings.Contains(extDef))
                {
                    TypeBinding binding = app.Create.NewTypeBinding(categories);
                    doc.ParameterBindings.Insert(extDef, binding, GroupTypeId.IdentityData);
                }
            }

            t.Commit();
        }
        finally
        {
            app.SharedParametersFilename = previousFile;
        }
    }

    /// <summary>Generates SharedParameters.txt from the GUID constants. Never call manually.</summary>
    public static void GenerateSharedParametersFile(string path)
    {
        var sb = new StringBuilder();
        sb.AppendLine("# This is a Revit shared parameter file.");
        sb.AppendLine("# Do not edit manually — regenerated from CoveninParameters.cs constants.");
        sb.AppendLine("*META\tVERSION\tMINVERSION");
        sb.AppendLine("META\t2\t1");
        sb.AppendLine("*GROUP\tID\tNAME");
        sb.AppendLine("GROUP\t1\t" + GroupName);
        sb.AppendLine("*PARAM\tGUID\tNAME\tDATATYPE\tDATACATEGORY\tGROUP\tVISIBLE\tDESCRIPTION\tUSERMODIFIABLE\tHIDEWHENNOVALUE");

        foreach (var (guid, name) in AllParams)
            sb.AppendLine($"PARAM\t{guid:D}\t{name}\tTEXT\t\t1\t1\t\t1\t0");

        File.WriteAllText(path, sb.ToString(), Encoding.UTF8);
    }

    private static string GetSharedParametersFilePath()
    {
        string dir = Path.GetDirectoryName(typeof(CoveninParameters).Assembly.Location)
                     ?? AppContext.BaseDirectory;
        return Path.Combine(dir, "SharedParameters.txt");
    }

    private static CategorySet BuildModelCategorySet(Document doc)
    {
        var set = new CategorySet();
        foreach (Category cat in doc.Settings.Categories)
        {
            if (cat.CategoryType == CategoryType.Model && cat.AllowsBoundParameters)
                set.Insert(cat);
        }
        return set;
    }
}
