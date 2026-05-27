namespace Ponchevit.Domain.Aliases;

public interface IAliasResolver
{
    string Resolve(string rawCode);
}

public class IdentityAliasResolver : IAliasResolver
{
    public string Resolve(string rawCode) => rawCode;
}
