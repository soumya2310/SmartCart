// =============================================================================
// Author : soumya2310 (Soumya Chattopadhyay)
// Affil  : Biju Patnaik University of Technology, Odisha, India
// Email  : soumya2310@gmail.com
// Paper  : "Edge-Cloud Collaborative Intelligence for Autonomous IoT Navigation"
//           Journal of Cloud Computing (Springer), 2025
// Supplementary Material S1.2
// CartTelemetryService.cs
// Cloud-Integrated AI-Driven Motorized Shopping Cart — Azure IoT Telemetry Layer
//
// Publishes 1 Hz cart-state payloads to Azure IoT Hub over MQTT/TLS.
// Integrates with .NET Generic Host for lifecycle management.
//
// Target:  .NET 8 LTS  |  C# 12
// NuGet:   Microsoft.Azure.Devices.Client (1.42.x)
//          Microsoft.Extensions.Hosting.Abstractions (8.x)
//          Microsoft.Extensions.Logging.Abstractions (8.x)
//          Microsoft.Extensions.Options (8.x)
// =============================================================================

using Microsoft.Azure.Devices.Client;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace SmartCart.Telemetry;

// ---------------------------------------------------------------------------
// Configuration (bound from appsettings.json)
// ---------------------------------------------------------------------------
public sealed class TelemetryOptions
{
    public string CartId               { get; init; } = string.Empty;
    public string IotHubConnectionString { get; init; } = string.Empty;
    public int    PublishIntervalMs    { get; init; } = 1_000;
}

// ---------------------------------------------------------------------------
// Cart state provided by the edge sensor-fusion layer
// ---------------------------------------------------------------------------
public interface ICartStateProvider
{
    Task<CartState> GetCurrentStateAsync(CancellationToken ct = default);
}

public record CartState(
    string CartId,
    double X,
    double Y,
    double BatterySoC,          // 0.0 – 1.0
    string? ActiveGoal,
    bool   AllSensorsHealthy,
    int    NavigationAccuracyPct);

// ---------------------------------------------------------------------------
// Telemetry payload — serialised to JSON before IoT Hub publish
// ---------------------------------------------------------------------------
public record TelemetryPayload(
    string CartId,
    long   TimestampUtcMs,
    double X,
    double Y,
    [property: JsonPropertyName("battery_soc")]  double BatterySoC,
    [property: JsonPropertyName("nav_goal")]     string NavigationGoal,
    [property: JsonPropertyName("sensors_ok")]   bool   SensorHealthOk,
    [property: JsonPropertyName("nav_accuracy")] int    NavigationAccuracyPct);

[JsonSerializable(typeof(TelemetryPayload))]
internal partial class TelemetryJsonContext : JsonSerializerContext { }

// ---------------------------------------------------------------------------
// Hosted service — runs as long as the .NET Generic Host is alive
// ---------------------------------------------------------------------------
public sealed class CartTelemetryService : IHostedService, IAsyncDisposable
{
    private readonly DeviceClient                    _iotClient;
    private readonly ICartStateProvider              _stateProvider;
    private readonly TelemetryOptions                _opts;
    private readonly ILogger<CartTelemetryService>   _log;
    private readonly PeriodicTimer                   _timer;

    private CancellationTokenSource? _cts;
    private Task?                    _loopTask;

    public CartTelemetryService(
        IOptions<TelemetryOptions>        options,
        ICartStateProvider                stateProvider,
        ILogger<CartTelemetryService>     log)
    {
        _opts          = options.Value;
        _stateProvider = stateProvider;
        _log           = log;
        _timer         = new PeriodicTimer(TimeSpan.FromMilliseconds(_opts.PublishIntervalMs));
        _iotClient     = DeviceClient.CreateFromConnectionString(
                             _opts.IotHubConnectionString,
                             TransportType.Mqtt);
    }

    // ── IHostedService ───────────────────────────────────────────────────────
    public Task StartAsync(CancellationToken ct)
    {
        _cts      = CancellationTokenSource.CreateLinkedTokenSource(ct);
        _loopTask = RunPublishLoopAsync(_cts.Token);
        _log.LogInformation("Cart telemetry publisher started (CartId={Id})", _opts.CartId);
        return Task.CompletedTask;
    }

    public async Task StopAsync(CancellationToken _)
    {
        if (_cts is not null) await _cts.CancelAsync();
        _log.LogInformation("Cart telemetry publisher stopping");
    }

    // ── Publish loop ─────────────────────────────────────────────────────────
    private async Task RunPublishLoopAsync(CancellationToken ct)
    {
        while (await _timer.WaitForNextTickAsync(ct))
        {
            CartState state;
            try
            {
                state = await _stateProvider.GetCurrentStateAsync(ct);
            }
            catch (Exception ex)
            {
                _log.LogWarning(ex, "Failed to retrieve cart state; skipping tick");
                continue;
            }

            var payload = new TelemetryPayload(
                CartId:               state.CartId,
                TimestampUtcMs:       DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(),
                X:                    state.X,
                Y:                    state.Y,
                BatterySoC:           Math.Round(state.BatterySoC, 3),
                NavigationGoal:       state.ActiveGoal ?? "IDLE",
                SensorHealthOk:       state.AllSensorsHealthy,
                NavigationAccuracyPct: state.NavigationAccuracyPct);

            var json    = JsonSerializer.Serialize(payload,
                              TelemetryJsonContext.Default.TelemetryPayload);
            var message = new Message(Encoding.UTF8.GetBytes(json))
            {
                ContentType     = "application/json",
                ContentEncoding = "utf-8",
                MessageId       = Guid.NewGuid().ToString()
            };
            message.Properties["cart-schema-version"] = "2.0";

            try
            {
                await _iotClient.SendEventAsync(message, ct);
                _log.LogDebug("Telemetry published: SoC={SoC:P0}  Sensors={Ok}",
                    state.BatterySoC, state.AllSensorsHealthy);
            }
            catch (Exception ex)
            {
                _log.LogError(ex, "IoT Hub publish failed; will retry next tick");
            }
        }
    }

    // ── IAsyncDisposable ─────────────────────────────────────────────────────
    public async ValueTask DisposeAsync()
    {
        _cts?.Dispose();
        _timer.Dispose();
        if (_loopTask is not null)
            await _loopTask.ConfigureAwait(false);
        await _iotClient.DisposeAsync();
    }
}
