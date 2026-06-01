using Ponchevit.Data;
using Ponchevit.Data.Sqlite;
using Ponchevit.Domain.Aliases;
using Ponchevit.Domain.Catalog;
using Ponchevit.Domain.Materials;
using Ponchevit.Infrastructure;
using Ponchevit.Revit.Context;
using Ponchevit.Revit.Families;
using Ponchevit.Revit.Materials;

namespace Ponchevit.Composition;

/// <summary>
/// Manual composition root. Built once in App.OnStartup; consumed via App.Services.
/// Commands call RevitContext.Attach(uiApp) before creating windows.
/// </summary>
public sealed class Services
{
    public ILog Log { get; }
    public RevitContextImpl RevitContext { get; }

    public IPartidasRepository PartidasRepository { get; }
    public ICoveninRulesRepository CoveninRulesRepository { get; }
    public PartidaCatalog PartidaCatalog { get; }
    public IAliasResolver AliasResolver { get; }

    public IMaterialMappingRepository MaterialMappingRepository { get; }
    public IMaterialMappingResolver MaterialMappingResolver { get; }

    public IFamilyGenerator[] FamilyGenerators { get; }

    private Services(
        ILog log,
        RevitContextImpl revitContext,
        IPartidasRepository partidasRepository,
        ICoveninRulesRepository coveninRulesRepository,
        PartidaCatalog partidaCatalog,
        IAliasResolver aliasResolver,
        IMaterialMappingRepository materialMappingRepository,
        IMaterialMappingResolver materialMappingResolver,
        IFamilyGenerator[] familyGenerators)
    {
        Log = log;
        RevitContext = revitContext;
        PartidasRepository = partidasRepository;
        CoveninRulesRepository = coveninRulesRepository;
        PartidaCatalog = partidaCatalog;
        AliasResolver = aliasResolver;
        MaterialMappingRepository = materialMappingRepository;
        MaterialMappingResolver = materialMappingResolver;
        FamilyGenerators = familyGenerators;
    }

    public static Services Build()
    {
        var log = new FileLog();

        var revitContext = new RevitContextImpl();

        var connectionFactory = new ConnectionFactory();
        var partidasRepo = new SqlitePartidasRepository(connectionFactory);
        var coveninRepo = new SqliteCoveninRulesRepository(connectionFactory);
        var catalog = new PartidaCatalog(partidasRepo, log);
        var aliasResolver = new IdentityAliasResolver();

        var materialMappingRepo = new ExtensibleStorageMaterialMappingRepository(
            () => revitContext.ActiveUiDocument.Document);
        var materialResolver = new MaterialMappingResolver(materialMappingRepo);

        var generators = new IFamilyGenerator[] { new MuroGenerator() };

        return new Services(
            log,
            revitContext,
            partidasRepo,
            coveninRepo,
            catalog,
            aliasResolver,
            materialMappingRepo,
            materialResolver,
            generators);
    }
}
