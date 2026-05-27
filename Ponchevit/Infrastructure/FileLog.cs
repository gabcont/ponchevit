using System;
using System.IO;

namespace Ponchevit.Infrastructure;

public sealed class FileLog : ILog
{
    private readonly string _logPath;
    private readonly object _gate = new();

    public FileLog()
    {
        string appData = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
        string folder = Path.Combine(appData, "Ponchevit");
        _logPath = Path.Combine(folder, "log.txt");

        try
        {
            if (!Directory.Exists(folder))
            {
                Directory.CreateDirectory(folder);
            }
        }
        catch
        {
            // Swallow IO exceptions
        }
    }

    public void Info(string message) => Write("INFO", message);
    public void Warn(string message) => Write("WARN", message);
    public void Error(string message, Exception? ex = null) => Write("ERROR", message, ex);

    private void Write(string level, string message, Exception? ex = null)
    {
        try
        {
            lock (_gate)
            {
                using var writer = new StreamWriter(_logPath, append: true);
                writer.WriteLine($"[{DateTime.Now:yyyy-MM-dd HH:mm:ss}] [{level}] {message}");
                if (ex != null)
                {
                    writer.WriteLine($"    Exception: {ex}");
                }
            }
        }
        catch
        {
            // Swallow IO exceptions
        }
    }
}
