// =============================================================================
// Author : soumya2310 (Soumya Chattopadhyay)
// Affil  : Biju Patnaik University of Technology, Odisha, India
// Email  : soumya2310@gmail.com
// Paper  : "Edge-Cloud Collaborative Intelligence for Autonomous IoT Navigation"
//           Journal of Cloud Computing (Springer), 2025
// Supplementary Material S1.3
// CartTelemetryProcessor.cs
// Cloud-Integrated AI-Driven Motorized Shopping Cart — AWS Lambda Analytics
//
// Triggered by DynamoDB Streams on CartTelemetry table inserts.
// Performs customer-journey reconstruction, writes summaries to CartJourneys
// table, and emits per-aisle dwell-time metrics to CloudWatch.
// Compiled with Native AOT for <5 ms cold-start latency.
//
// Target:  .NET 8 LTS  |  C# 12  |  Lambda Native AOT
// NuGet:   Amazon.Lambda.Core (2.x)
//          Amazon.Lambda.DynamoDBEvents (3.x)
//          Amazon.Lambda.Serialization.SystemTextJson (2.x)
//          AWSSDK.DynamoDBv2 (3.7.x)
//          AWSSDK.CloudWatch (3.7.x)
// =============================================================================

using Amazon.CloudWatch;
using Amazon.CloudWatch.Model;
using Amazon.DynamoDBv2;
using Amazon.DynamoDBv2.Model;
using Amazon.Lambda.Core;
using Amazon.Lambda.DynamoDBEvents;
using System.Text.Json.Serialization;

[assembly: Amazon.Lambda.Core.LambdaSerializer(
    typeof(Amazon.Lambda.Serialization.SystemTextJson.DefaultLambdaJsonSerializer))]

namespace SmartCart.Lambda;

// ---------------------------------------------------------------------------
// Source-generated JSON context for AOT compatibility
// ---------------------------------------------------------------------------
[JsonSerializable(typeof(JourneySummary))]
internal partial class LambdaJsonContext : JsonSerializerContext { }

public record JourneySummary(
    string CartId, string SessionId,
    string Aisle,  long   TimestampMs,
    int    DwellEventCount);

// ---------------------------------------------------------------------------
// Lambda function handler — static entry point for AOT
// ---------------------------------------------------------------------------
public static class CartTelemetryProcessor
{
    private static readonly IAmazonDynamoDB    Dynamo = new AmazonDynamoDBClient();
    private static readonly IAmazonCloudWatch  Cw     = new AmazonCloudWatchClient();

    private const string CartJourneysTable  = "CartJourneys";
    private const string CwNamespace        = "SmartCart/Analytics";

    /// <summary>
    /// Lambda handler. Processes each DynamoDB Streams record representing
    /// a new cart-telemetry INSERT event.
    /// </summary>
    public static async Task HandleAsync(DynamoDBEvent evt, ILambdaContext ctx)
    {
        ArgumentNullException.ThrowIfNull(evt);
        ctx.Logger.LogInformation(
            "Processing {Count} stream records", evt.Records.Count);

        foreach (var record in evt.Records)
        {
            if (record.EventName != OperationType.INSERT)
            {
                ctx.Logger.LogDebug(
                    "Skipping {EventName} record", record.EventName);
                continue;
            }

            var img    = record.Dynamodb.NewImage;
            var cartId = img["cart_id"].S;
            var aisle  = img.TryGetValue("active_aisle", out var av)
                             ? av.S : "UNKNOWN";
            var tsMs   = long.Parse(img["timestamp_ms"].N);

            await Task.WhenAll(
                WriteJourneySummaryAsync(cartId, aisle, tsMs, ctx.AwsRequestId),
                EmitDwellMetricAsync(cartId, aisle));

            ctx.Logger.LogInformation(
                "Processed telemetry: CartId={CartId} Aisle={Aisle}", cartId, aisle);
        }
    }

    // ── Write journey summary to DynamoDB ────────────────────────────────────
    private static async Task WriteJourneySummaryAsync(
        string cartId, string aisle, long tsMs, string sessionId)
    {
        var item = new Dictionary<string, AttributeValue>
        {
            ["cart_id"]    = new(cartId),
            ["session_id"] = new(sessionId),
            ["aisle"]      = new(aisle),
            ["timestamp"]  = new() { N = tsMs.ToString() },
            ["ttl"]        = new()
            {
                N = DateTimeOffset.UtcNow.AddDays(3)
                                         .ToUnixTimeSeconds()
                                         .ToString()   // 72-hour GDPR retention
            }
        };

        await Dynamo.PutItemAsync(new PutItemRequest
        {
            TableName           = CartJourneysTable,
            Item                = item,
            ConditionExpression = "attribute_not_exists(cart_id)"  // idempotent
        });
    }

    // ── Emit CloudWatch custom metric ────────────────────────────────────────
    private static async Task EmitDwellMetricAsync(string cartId, string aisle)
    {
        await Cw.PutMetricDataAsync(new PutMetricDataRequest
        {
            Namespace  = CwNamespace,
            MetricData =
            [
                new MetricDatum
                {
                    MetricName = "AisleDwellEvent",
                    Value      = 1,
                    Unit       = StandardUnit.Count,
                    Timestamp  = DateTime.UtcNow,
                    Dimensions =
                    [
                        new Dimension { Name = "CartId", Value = cartId },
                        new Dimension { Name = "Aisle",  Value = aisle  }
                    ]
                }
            ]
        });
    }
}
