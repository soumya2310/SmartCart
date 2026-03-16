// =============================================================================
// Supplementary Material S3.1
// GcpCloudFunctionsTelemetryProcessor.cs
// Cloud-Integrated AI-Driven Motorized Shopping Cart — GCP Cloud Functions Analytics
//
// Author : soumya2310 (Soumya Chattopadhyay)
// Affil  : Biju Patnaik University of Technology, Odisha, India
// Email  : soumya2310@gmail.com
// Paper  : "Edge-Cloud Collaborative Intelligence for Autonomous IoT Navigation"
//           Journal of Cloud Computing (Springer), 2025
//
// HTTP-triggered Cloud Function that receives cart telemetry events forwarded
// from Cloud IoT Core via Cloud Pub/Sub push subscriptions.
// Performs customer-journey reconstruction, writes summaries to Firestore,
// and emits dwell-time metrics to Cloud Monitoring via the Google.Cloud.Monitoring API.
//
// Target:  .NET 8 LTS  |  C# 12  |  Google Cloud Functions Framework
// NuGet:   Google.Cloud.Functions.Framework (2.2.x)
//          Google.Cloud.Firestore (3.7.x)
//          Google.Cloud.Monitoring.V3 (3.9.x)
//          Google.Api.CommonProtos (2.15.x)   ← provides Google.Api.Metric / MonitoredResource
//          Google.Api.Gax (4.9.x)             ← provides ProjectName resource-name helper
//          Google.Protobuf (3.27.x)
//
// FIXES APPLIED
// ─────────────────────────────────────────────────────────────────────────────
// CS0246 line 204 — 'ProjectName' not found
//   Root cause: missing "using Google.Api.Gax.ResourceNames;" — ProjectName is
//   defined in Google.Api.Gax (resource-name helper types).
//   Fix: using Google.Api.Gax.ResourceNames; added below.
//
// CS0246 line 209 — 'Metric' not found
// CS0246 line 214 — 'MonitoredResource' not found
//   Root cause: both types live in the Google.Api namespace (package
//   Google.Api.CommonProtos). The using directive was missing.
//   Fix: using Google.Api; added below.
//
// CS0029 line 223 — cannot convert Google.Cloud.Firestore.Timestamp
//                   to Google.Protobuf.WellKnownTypes.Timestamp
//   Root cause: both Google.Cloud.Firestore and Google.Protobuf.WellKnownTypes
//   expose a type named Timestamp. When the compiler resolves the bare name
//   'Timestamp' it prefers Google.Cloud.Firestore.Timestamp (imported via
//   Google.Cloud.Firestore), which cannot be implicitly cast to the Protobuf
//   Timestamp expected by TimeInterval.EndTime.
//   Fix: aliased the Protobuf type as ProtoTimestamp and used that alias
//   everywhere a Google.Protobuf.WellKnownTypes.Timestamp is required.
// =============================================================================

// FIX CS0246 (Metric, MonitoredResource) — types are in the Google.Api namespace
using Google.Api;

// FIX CS0246 (ProjectName) — resource-name helper lives in Google.Api.Gax.ResourceNames
using Google.Api.Gax.ResourceNames;

using Google.Cloud.Firestore;
using Google.Cloud.Functions.Framework;
using Google.Cloud.Monitoring.V3;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using System.Text.Json;
using System.Text.Json.Serialization;

// FIX CS0029 — alias the Protobuf Timestamp to avoid ambiguity with
// Google.Cloud.Firestore.Timestamp (both are imported above).
// Use ProtoTimestamp wherever Google.Protobuf.WellKnownTypes.Timestamp is needed.
using ProtoTimestamp = Google.Protobuf.WellKnownTypes.Timestamp;

namespace SmartCart.Gcp.Functions;

// ---------------------------------------------------------------------------
// Pub/Sub push message envelope (Cloud IoT Core → Pub/Sub → Cloud Functions)
// ---------------------------------------------------------------------------
public sealed class PubSubEnvelope
{
    [JsonPropertyName("message")]
    public PubSubMessage? Message { get; init; }
}

public sealed class PubSubMessage
{
    [JsonPropertyName("data")]
    public string? Data { get; init; }           // base64-encoded JSON

    [JsonPropertyName("messageId")]
    public string? MessageId { get; init; }

    [JsonPropertyName("publishTime")]
    public string? PublishTime { get; init; }
}

// ---------------------------------------------------------------------------
// Domain models
// ---------------------------------------------------------------------------
[FirestoreData]
public class JourneySummaryGcp
{
    [FirestoreProperty] public string CartId           { get; set; } = string.Empty;
    [FirestoreProperty] public string SessionId        { get; set; } = string.Empty;
    [FirestoreProperty] public string Aisle            { get; set; } = string.Empty;
    [FirestoreProperty] public long   EntryTimestampMs { get; set; }
    [FirestoreProperty] public long   ExitTimestampMs  { get; set; }

    // Computed — not stored in Firestore (no FirestoreProperty attribute)
    public double DwellSeconds => (ExitTimestampMs - EntryTimestampMs) / 1_000.0;
}

public record CartTelemetryPayload(
    string  CartId,
    string  SessionId,
    double  X,
    double  Y,
    float   BatterySoc,
    string? CurrentAisle,
    long    TimestampMs);

// ---------------------------------------------------------------------------
// Cloud Function — implements IHttpFunction for Pub/Sub push endpoint
// ---------------------------------------------------------------------------
public sealed class GcpTelemetryProcessor : IHttpFunction
{
    private const string ProjectId    = "smart-cart-prod";
    private const string CollectionId = "cart_journeys";
    private const double MaxDwellSec  = 900.0;

    private readonly FirestoreDb                     _firestore;
    private readonly MetricServiceClient             _metrics;
    private readonly ILogger<GcpTelemetryProcessor>  _log;

    // Session state — GCP Cloud Functions can be long-lived; state persists
    // within a single function instance. For cross-instance coordination, use
    // Cloud Memorystore (Redis) in production deployments > 100 simultaneous carts.
    private readonly Dictionary<string, (string Aisle, long EntryMs)> _sessions = new();

    public GcpTelemetryProcessor(
        FirestoreDb firestore,
        MetricServiceClient metrics,
        ILogger<GcpTelemetryProcessor> log)
    {
        _firestore = firestore;
        _metrics   = metrics;
        _log       = log;
    }

    /// <summary>
    /// Handles Pub/Sub push POST from Cloud IoT Core telemetry subscription.
    /// </summary>
    public async Task HandleAsync(HttpContext context)
    {
        PubSubEnvelope? envelope;
        try
        {
            envelope = await JsonSerializer.DeserializeAsync<PubSubEnvelope>(
                context.Request.Body);
        }
        catch (JsonException ex)
        {
            _log.LogError(ex, "Failed to deserialize Pub/Sub envelope");
            context.Response.StatusCode = 400;
            return;
        }

        if (envelope?.Message?.Data is null)
        {
            _log.LogWarning("Empty Pub/Sub message received");
            context.Response.StatusCode = 204;
            return;
        }

        // Decode base64 payload
        byte[] rawJson = Convert.FromBase64String(envelope.Message.Data);
        CartTelemetryPayload? payload;
        try
        {
            payload = JsonSerializer.Deserialize<CartTelemetryPayload>(rawJson);
        }
        catch (JsonException ex)
        {
            _log.LogError(ex, "Failed to deserialize cart telemetry payload");
            context.Response.StatusCode = 400;
            return;
        }

        if (payload is null)
        {
            context.Response.StatusCode = 204;
            return;
        }

        var summary = ProcessEvent(payload);
        if (summary is not null)
        {
            await Task.WhenAll(
                WriteToFirestoreAsync(summary),
                EmitDwellMetricAsync(summary));

            _log.LogInformation(
                "Journey summary written: Cart={CartId} Aisle={Aisle} Dwell={Dwell:F1}s",
                summary.CartId, summary.Aisle, summary.DwellSeconds);
        }

        context.Response.StatusCode = 200;
    }

    private JourneySummaryGcp? ProcessEvent(CartTelemetryPayload payload)
    {
        if (string.IsNullOrEmpty(payload.CurrentAisle))
            return null;

        var key = $"{payload.CartId}:{payload.SessionId}";

        if (_sessions.TryGetValue(key, out var current) &&
            current.Aisle != payload.CurrentAisle)
        {
            var dwell = (payload.TimestampMs - current.EntryMs) / 1_000.0;
            _sessions[key] = (payload.CurrentAisle, payload.TimestampMs);

            if (dwell is > 1.0 and < MaxDwellSec)
            {
                return new JourneySummaryGcp
                {
                    CartId           = payload.CartId,
                    SessionId        = payload.SessionId,
                    Aisle            = current.Aisle,
                    EntryTimestampMs = current.EntryMs,
                    ExitTimestampMs  = payload.TimestampMs
                };
            }
        }
        else if (!_sessions.ContainsKey(key))
        {
            _sessions[key] = (payload.CurrentAisle, payload.TimestampMs);
        }

        return null;
    }

    private async Task WriteToFirestoreAsync(JourneySummaryGcp summary)
    {
        var docRef = _firestore
            .Collection(CollectionId)
            .Document($"{summary.CartId}_{summary.SessionId}_{summary.EntryTimestampMs}");

        await docRef.SetAsync(summary);
    }

    private async Task EmitDwellMetricAsync(JourneySummaryGcp summary)
    {
        // FIX CS0246 (line 204) — ProjectName
        // Before: new ProjectName(ProjectId)  ← type not found without the ResourceNames using
        // After:  ProjectName.FromProject(ProjectId) from Google.Api.Gax.ResourceNames
        var projectName = ProjectName.FromProject(ProjectId);

        // FIX CS0029 (line 223) — Timestamp ambiguity
        // Before: var now = Timestamp.FromDateTime(DateTime.UtcNow);
        //   The compiler resolved 'Timestamp' to Google.Cloud.Firestore.Timestamp
        //   (also in scope), then could not assign it to TimeInterval.EndTime
        //   which expects Google.Protobuf.WellKnownTypes.Timestamp.
        // After:  use the ProtoTimestamp alias declared at the top of this file.
        ProtoTimestamp now = ProtoTimestamp.FromDateTime(DateTime.UtcNow);

        // FIX CS0246 (line 209) — Metric
        // 'Metric' is Google.Api.Metric (from Google.Api.CommonProtos).
        // Resolved by: using Google.Api; at the top of this file.
        var metric = new Metric
        {
            Type   = "custom.googleapis.com/smart_cart/aisle_dwell_seconds",
            Labels = { ["cart_id"] = summary.CartId, ["aisle"] = summary.Aisle }
        };

        // FIX CS0246 (line 214) — MonitoredResource
        // 'MonitoredResource' is Google.Api.MonitoredResource (same package).
        // Resolved by the same: using Google.Api;
        var resource = new MonitoredResource
        {
            Type   = "global",
            Labels = { ["project_id"] = ProjectId }
        };

        var timeSeries = new TimeSeries
        {
            Metric   = metric,
            Resource = resource,
            Points   =
            {
                new Point
                {
                    Interval = new TimeInterval { EndTime = now },
                    Value    = new TypedValue { DoubleValue = summary.DwellSeconds }
                }
            }
        };

        await _metrics.CreateTimeSeriesAsync(projectName, new[] { timeSeries });
    }
}
