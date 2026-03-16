// =============================================================================
// Author : soumya2310 (Soumya Chattopadhyay)
// Affil  : Biju Patnaik University of Technology, Odisha, India
// Email  : soumya2310@gmail.com
// Paper  : "Edge-Cloud Collaborative Intelligence for Autonomous IoT Navigation"
//           Journal of Cloud Computing (Springer), 2025
// Supplementary Material S2.1
// AzureFunctionsTelemetryProcessor.cs
// Cloud-Integrated AI-Driven Motorized Shopping Cart — Azure Functions Analytics
//
// Triggered by Azure IoT Hub Event Hub-compatible endpoint via EventHubTrigger.
// Performs customer-journey reconstruction, writes summaries to Azure Cosmos DB,
// and emits per-aisle dwell-time metrics to Azure Monitor.
// Implemented using Azure Functions v4 isolated worker process model (.NET 8).
//
// Target:  .NET 8 LTS  |  C# 12  |  Azure Functions v4 Isolated Worker
// NuGet:   Microsoft.Azure.Functions.Worker (1.22.x)
//          Microsoft.Azure.Functions.Worker.Extensions.EventHubs (6.x)
//          Microsoft.Azure.Cosmos (3.38.x)
//          Azure.Monitor.OpenTelemetry.AspNetCore (1.2.x)
//          Microsoft.Extensions.Logging.Abstractions (8.x)
// =============================================================================

using Azure.Messaging.EventHubs;
using Microsoft.Azure.Cosmos;
using Microsoft.Azure.Functions.Worker;
using Microsoft.Extensions.Logging;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace SmartCart.Azure.Functions;

// ---------------------------------------------------------------------------
// Domain models
// ---------------------------------------------------------------------------
public record CartTelemetryEvent(
    string  CartId,
    string  SessionId,
    double  X,
    double  Y,
    float   BatterySoc,
    string? CurrentAisle,
    long    TimestampMs);

public record JourneySummary(
    string CartId,
    string SessionId,
    string Aisle,
    long   EntryTimestampMs,
    long   ExitTimestampMs)
{
    public double DwellSeconds => (ExitTimestampMs - EntryTimestampMs) / 1_000.0;
    public string Id => $"{CartId}:{SessionId}:{Aisle}:{EntryTimestampMs}";
}

// ---------------------------------------------------------------------------
// Azure Function implementation
// ---------------------------------------------------------------------------
public sealed class AzureTelemetryProcessor
{
    private const string DatabaseId   = "SmartCartDb";
    private const string ContainerId  = "CartJourneys";
    private const double MaxDwellSec  = 900.0;  // discard stale sessions > 15 min

    private readonly CosmosClient                    _cosmos;
    private readonly ILogger<AzureTelemetryProcessor> _log;

    // In-memory session state — in production, use Azure Cache for Redis
    // for multi-instance fanout scenarios (> 50 concurrent carts per instance)
    private readonly Dictionary<string, (string Aisle, long EntryMs)> _sessions = new();

    public AzureTelemetryProcessor(CosmosClient cosmos,
                                   ILogger<AzureTelemetryProcessor> log)
    {
        _cosmos = cosmos;
        _log    = log;
    }

    /// <summary>
    /// Processes a batch of IoT Hub telemetry events from the Event Hub endpoint.
    /// Azure Functions batches multiple events per invocation for throughput.
    /// </summary>
    [Function("CartTelemetryProcessor")]
    public async Task RunAsync(
        [EventHubTrigger("cart-telemetry", Connection = "IoTHubEventHubConn")]
        EventData[] events,
        FunctionContext context,
        CancellationToken ct)
    {
        _log.LogInformation("Processing batch of {Count} telemetry events", events.Length);

        var container = _cosmos.GetContainer(DatabaseId, ContainerId);
        var batch     = new List<JourneySummary>(events.Length);

        foreach (var ev in events)
        {
            CartTelemetryEvent? telemetry;
            try
            {
                telemetry = JsonSerializer.Deserialize<CartTelemetryEvent>(
                    ev.EventBody.ToString());

                if (telemetry is null)
                {
                    _log.LogWarning("Null telemetry payload — skipping");
                    continue;
                }
            }
            catch (JsonException ex)
            {
                _log.LogError(ex, "Failed to deserialize telemetry event body");
                continue;
            }

            var summary = ProcessTelemetryEvent(telemetry);
            if (summary is not null)
                batch.Add(summary);
        }

        // Bulk-write completed journey summaries to Cosmos DB
        var tasks = batch.Select(s => UpsertJourneyAsync(container, s, ct));
        await Task.WhenAll(tasks);

        _log.LogInformation("Wrote {Count} journey summaries to Cosmos DB", batch.Count);
    }

    private JourneySummary? ProcessTelemetryEvent(CartTelemetryEvent ev)
    {
        var key = $"{ev.CartId}:{ev.SessionId}";

        if (string.IsNullOrEmpty(ev.CurrentAisle))
            return null;

        if (_sessions.TryGetValue(key, out var current))
        {
            // Aisle transition — finalise previous dwell period
            if (current.Aisle != ev.CurrentAisle)
            {
                var dwell = (ev.TimestampMs - current.EntryMs) / 1_000.0;
                _sessions[key] = (ev.CurrentAisle, ev.TimestampMs);

                if (dwell < MaxDwellSec && dwell > 1.0)
                {
                    return new JourneySummary(
                        CartId:           ev.CartId,
                        SessionId:        ev.SessionId,
                        Aisle:            current.Aisle,
                        EntryTimestampMs: current.EntryMs,
                        ExitTimestampMs:  ev.TimestampMs);
                }
            }
        }
        else
        {
            _sessions[key] = (ev.CurrentAisle, ev.TimestampMs);
        }

        return null;
    }

    private static async Task UpsertJourneyAsync(
        Container container, JourneySummary summary, CancellationToken ct)
    {
        await container.UpsertItemAsync(
            item:            summary,
            partitionKey:    new PartitionKey(summary.CartId),
            cancellationToken: ct);
    }
}
