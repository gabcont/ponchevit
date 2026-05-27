using Xunit;
using Ponchevit.Domain.Graph;
using Ponchevit.Domain.Model;
using System.Collections.Generic;

namespace Ponchevit.Tests.Domain.Graph;

public class CodeAssemblerTests
{
    private readonly CodeAssembler _assembler = new();

    [Fact]
    public void Assemble_ConcatenatesCodesCorrectly()
    {
        var path = new List<Conexion>
        {
            new Conexion("C1", null, "E4", "COL1", null),
            new Conexion("C2", "C1", "1", "COL2", "V1"),
            new Conexion("C3", "C2", "2", "COL3", "V2")
        };

        var result = _assembler.Assemble(path);

        Assert.Equal("E412", result.Value);
    }

    [Fact]
    public void Assemble_HandlesEmptyBridges()
    {
        var path = new List<Conexion>
        {
            new Conexion("C1", null, "E4", "COL1", null),
            new Conexion("C2", "C1", "", "COL2", "V1"), // Bridge
            new Conexion("C3", "C2", "1", "COL3", "V2")
        };

        var result = _assembler.Assemble(path);

        Assert.Equal("E41", result.Value);
    }

    [Fact]
    public void Assemble_RespectsTenDigitCap()
    {
        var path = new List<Conexion>
        {
            new Conexion("C1", null, "E4", "COL1", null),      // 2
            new Conexion("C2", "C1", "123", "COL2", "V1"),   // 5
            new Conexion("C3", "C2", "456", "COL3", "V2"),   // 8
            new Conexion("C4", "C3", "789", "COL4", "V3")    // 11 -> Should cap at 10
        };

        var result = _assembler.Assemble(path);

        Assert.Equal(10, result.Length);
        Assert.Equal("E412345678", result.Value);
    }

    [Fact]
    public void ComputePrefix_ReturnsPartialCode()
    {
        var path = new List<Conexion>
        {
            new Conexion("C1", null, "E4", "COL1", null),
            new Conexion("C2", "C1", "1", "COL2", "V1"),
            new Conexion("C3", "C2", "2", "COL3", "V2")
        };

        var result = _assembler.ComputePrefix(path, "C2");

        Assert.Equal("E41", result.Value);
    }

    [Fact]
    public void ComputePrefix_ReturnsFullCode_IfIdAtEnd()
    {
        var path = new List<Conexion>
        {
            new Conexion("C1", null, "E4", "COL1", null),
            new Conexion("C2", "C1", "1", "COL2", "V1")
        };

        var result = _assembler.ComputePrefix(path, "C2");

        Assert.Equal("E41", result.Value);
    }

    [Fact]
    public void ComputePrefix_ReturnsEmpty_IfIdNotFound()
    {
        var path = new List<Conexion>
        {
            new Conexion("C1", null, "E4", "COL1", null)
        };

        var result = _assembler.ComputePrefix(path, "UNKNOWN");

        Assert.True(result.IsEmpty);
        Assert.Equal(string.Empty, result.Value);
    }
}
