using Ponchevit.Infrastructure;

namespace Ponchevit.Composition;

/// <summary>
/// Manual composition root. Holds shared singletons used across commands and UI.
/// Built once in App.OnStartup; consumed via App.Services.
/// Will grow in Phase 1 (repositories, catalog) and Phase 2 (IRevitContext).
/// </summary>
public sealed class Services
{
    public ILog Log { get; }

    private Services(ILog log)
    {
        Log = log;
    }

    public static Services Build()
    {
        var log = new FileLog();
        return new Services(log);
    }
}
