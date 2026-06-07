using Ponchevit.Data;
using Ponchevit.Data.Sqlite;
using Ponchevit.Domain.Aliases;
using Ponchevit.Domain.Catalog;
using Ponchevit.Domain.Materials;
using Ponchevit.Infrastructure;
using Ponchevit.Revit;
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

    // Fix #10: PartidaConstructibilityResolver is still built eagerly at startup
    // (one-time DFS over the DAG with prefix-pruning; fast enough on commodity hardware).
    // Decision: keep eager construction because:
    //   (a) the resolver is always needed on first window open, so lazy adds no benefit,
    //   (b) the DFS is bounded by the prefix-pruning index and completes in < 1s,
    //   (c) deferring into Task.Run would require synchronization on first access with
    //       no user-visible indicator — worse UX than a slightly longer startup.
    // See ADR 2026-06-06 — Constructibility eager-vs-lazy decision.
    public PartidaConstructibilityResolver ConstructibilityResolver { get; }

    public IAliasResolver AliasResolver { get; }

    public IMaterialMappingRepository MaterialMappingRepository { get; }
    public IMaterialMappingResolver MaterialMappingResolver { get; }

    public IFamilyGenerator[] FamilyGenerators { get; }

    /// <summary>
    /// Shared resolver for Partida → Sección/Subcapítulo/Capítulo hierarchy.
    /// Built once from the eagerly-loaded catalog tables; deterministic for the session.
    /// Fix C: injected into AgregarFamiliaViewModel so BuildGeneratorInput does not
    /// reconstruct it on every Agregar click.
    /// </summary>
    public PartidaHierarchyResolver HierarchyResolver { get; }

    /// <summary>
    /// Orchestrates the two-transaction element generation flow (bind params, then create).
    /// Consumed by AgregarFamiliaViewModel via PostExternalEvent.
    /// Fix #1: Transaction logic moved out of the VM and into this Revit-layer class.
    /// </summary>
    public FamilyGenerationOrchestrator GenerationOrchestrator { get; }

    /// <summary>
    /// Returns project materials from a Revit Document.
    /// Consumed by both AgregarFamiliaCommand and MapeoMaterialesCommand.
    /// Fix #6 (option B): centralises FilteredElementCollector use; both commands share
    /// the same service; the window code-behind no longer touches RevitAPI.
    /// See ADR 2026-06-06 — Project-material query service.
    /// </summary>
    public IProjectMaterialQuery ProjectMaterialQuery { get; }

    private Services(
        ILog log,
        RevitContextImpl revitContext,
        IPartidasRepository partidasRepository,
        ICoveninRulesRepository coveninRulesRepository,
        PartidaCatalog partidaCatalog,
        PartidaConstructibilityResolver constructibilityResolver,
        IAliasResolver aliasResolver,
        IMaterialMappingRepository materialMappingRepository,
        IMaterialMappingResolver materialMappingResolver,
        IFamilyGenerator[] familyGenerators,
        FamilyGenerationOrchestrator generationOrchestrator,
        IProjectMaterialQuery projectMaterialQuery,
        PartidaHierarchyResolver hierarchyResolver)
    {
        Log = log;
        RevitContext = revitContext;
        PartidasRepository = partidasRepository;
        CoveninRulesRepository = coveninRulesRepository;
        PartidaCatalog = partidaCatalog;
        ConstructibilityResolver = constructibilityResolver;
        AliasResolver = aliasResolver;
        MaterialMappingRepository = materialMappingRepository;
        MaterialMappingResolver = materialMappingResolver;
        FamilyGenerators = familyGenerators;
        GenerationOrchestrator = generationOrchestrator;
        ProjectMaterialQuery = projectMaterialQuery;
        HierarchyResolver = hierarchyResolver;
    }

    public static Services Build()
    {
        var log = new FileLog();

        var revitContext = new RevitContextImpl();

        var connectionFactory = new ConnectionFactory();
        var partidasRepo = new SqlitePartidasRepository(connectionFactory);
        var coveninRepo = new SqliteCoveninRulesRepository(connectionFactory);
        var catalog = new PartidaCatalog(partidasRepo, log);
        var constructibilityResolver = new PartidaConstructibilityResolver(coveninRepo, catalog.GetPartidas());
        var aliasResolver = new IdentityAliasResolver();

        var materialMappingRepo = new ExtensibleStorageMaterialMappingRepository(
            () => revitContext.ActiveUiDocument.Document);
        var materialResolver = new MaterialMappingResolver(materialMappingRepo);

        // ── Element modules ────────────────────────────────────────────────────
        // To add a new element type (e.g. PuertaGenerator, VentanaGenerator):
        //   1. Create a class in Revit/Families/ implementing IFamilyGenerator.
        //      Use MuroGenerator as the reference implementation.
        //   2. Add a recognizer class in Revit/Families/ implementing IElementRecognizer
        //      (Phase 5+). Use MuroRecognizer as the reference implementation.
        //   3. Register both instances in the arrays below.
        //   4. For elements that need a base RFA, ship it in Resources/Families/
        //      and load it in the generator via Document.LoadFamily before Generate().
        var generators = new IFamilyGenerator[] { new MuroGenerator(materialMappingRepo) };
        // var recognizers = new IElementRecognizer[] { new MuroRecognizer(materialMappingRepo) }; // Phase 5
        var orchestrator = new FamilyGenerationOrchestrator(log);
        var materialQuery = new ProjectMaterialQuery(revitContext);

        // Fix C: build the resolver once; inject into the VM so BuildGeneratorInput
        // does not reconstruct it on every Agregar click.
        var hierarchyResolver = new PartidaHierarchyResolver(
            partidasRepo.GetCapitulos(),
            partidasRepo.GetSubcapitulos(),
            partidasRepo.GetSecciones());

        return new Services(
            log,
            revitContext,
            partidasRepo,
            coveninRepo,
            catalog,
            constructibilityResolver,
            aliasResolver,
            materialMappingRepo,
            materialResolver,
            generators,
            orchestrator,
            materialQuery,
            hierarchyResolver);
    }
}
