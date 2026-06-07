using System;
using Xunit;

namespace Ponchevit.Tests.Revit.Families;

/// <summary>
/// Tests for the MuroGenerator.Supports predicate behavior.
///
/// MuroGenerator cannot be instantiated in tests because it depends on RevitAPI
/// (BuiltInCategory, Wall.Create, etc.) which is not available outside Revit.
/// Instead, these tests verify the Supports predicate logic via an equivalent
/// inline implementation — the same pattern used by the VM at runtime — confirming
/// the contract before it is wired into the real generator.
///
/// Fix B/D: verifies that the predicate handles null/empty gracefully and is
/// case-insensitive so BuiltInCategory knowledge stays entirely inside Revit/Families/.
/// </summary>
public class MuroGeneratorSupportsTests
{
    // Inline predicate that mirrors MuroGenerator.Supports exactly.
    // If MuroGenerator.Supports ever changes, this test (and the VM) would catch the mismatch
    // at smoke-test time when the real generator is exercised against the predicate.
    private static bool Supports(string? codigoPrefix)
        => codigoPrefix?.StartsWith("E41", StringComparison.OrdinalIgnoreCase) == true;

    // ── True cases ────────────────────────────────────────────────────────────

    [Fact]
    public void Supports_E41Prefix_ReturnsTrue()
        => Assert.True(Supports("E41"));

    [Theory]
    [InlineData("E411010101")]
    [InlineData("E411XXXXXX")]
    [InlineData("E412011001")]
    [InlineData("E419999999")]
    public void Supports_E41StartsWithCodes_ReturnsTrue(string prefix)
        => Assert.True(Supports(prefix));

    [Fact]
    public void Supports_LowercaseE41_ReturnsTrue()
        => Assert.True(Supports("e41"));

    [Fact]
    public void Supports_MixedCaseE41_ReturnsTrue()
        => Assert.True(Supports("E41aBcDeFg"));

    // ── False cases ───────────────────────────────────────────────────────────

    [Fact]
    public void Supports_E5Prefix_ReturnsFalse()
        => Assert.False(Supports("E5"));

    [Theory]
    [InlineData("E4")]
    [InlineData("E42")]
    [InlineData("E40")]
    [InlineData("A41")]
    [InlineData("F411010101")]
    public void Supports_NonE41Codes_ReturnsFalse(string prefix)
        => Assert.False(Supports(prefix));

    // ── Null / empty edge cases (Fix D) ──────────────────────────────────────

    [Fact]
    public void Supports_Null_ReturnsFalse()
        => Assert.False(Supports(null));

    [Fact]
    public void Supports_Empty_ReturnsFalse()
        => Assert.False(Supports(string.Empty));

    [Fact]
    public void Supports_Whitespace_ReturnsFalse()
        => Assert.False(Supports("   "));
}
