// =============================================================================
// Supplementary Material S1.5 (GCP portion — split from CloudFunctionsAll.cs)
// GcpFunctionsAll.cs
// Cloud-Integrated AI-Driven Motorized Shopping Cart — GCP Cloud Functions
//
// Author : soumya2310 (Soumya Chattopadhyay)
// Affil  : Biju Patnaik University of Technology, Odisha, India
// Email  : soumya2310@gmail.com
// Paper  : "Edge-Cloud Collaborative Intelligence for Autonomous IoT Navigation"
//           Journal of Cloud Computing (Springer), 2025
//
// GCP Cloud Function equivalent of the Azure functions in CloudFunctionsAll.cs.
// Triggered via HTTP by a Pub/Sub push subscription (Cloud IoT Core pipeline).
// Writes cart telemetry summaries to Firestore CartJourneys collection.
//
// Deployment:
//   gcloud functions deploy CartAnalyticsGcp \
//       --runtime dotnet8 --trigger-http --allow-unauthenticated
//
// Target:  .NET 8 LTS  |  C# 12
// NuGet:   Google.Cloud.Functions.Framework (2.x)
//          Google.Cloud.Firestore (3.x)
//
// NAMESPACE FIX
// ─────────────────────────────────────────────────────────────────────────────
// This file was extracted from the original CloudFunctionsAll.cs to resolve
// the CS8955 error caused by having two file-scoped namespace declarations
// (namespace SmartCart.Azure.Functions; and namespace SmartCart.GCP.Functions;)
// in the same file. One file-scoped namespace per file is the C# 10+ rule.
// =============================================================================

using Google.Cloud.Functions.Framework;
using Google.Cloud.Firestore;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using System.Text.Json;

namespace SmartCart.GCP.Functions;

/// <summary>
/// GCP Cloud Function: triggered by Pub/Sub message from IoT Core.
/// Writes cart telemetry summary to Firestore CartJourneys collection.
/// Deployed with: gcloud functions deploy CartAnalyticsGcp --runtime dotnet8
/// </summary>
public class CartAnalyticsGcp : IHttpFunction
{
    private readonly FirestoreDb              _db;
    private readonly ILogger<CartAnalyticsGcp> _log;

    public CartAnalyticsGcp(FirestoreDb db, ILogger<CartAnalyticsGcp> log)
    {
        _db  = db;
        _log = log;
    }

    public async Task HandleAsync(HttpContext ctx)
    {
        // Pub/Sub push subscription delivers base64-encoded JSON
        using var reader = new StreamReader(ctx.Request.Body);
        var body = await reader.ReadToEndAsync();

        // Decode Pub/Sub envelope
        using var doc = JsonDocument.Parse(body);
        var messageData = doc.RootElement
                             .GetProperty("message")
                             .GetProperty("data")
                             .GetString() ?? string.Empty;

        var json = System.Text.Encoding.UTF8.GetString(
                       Convert.FromBase64String(messageData));

        using var telemetry = JsonDocument.Parse(json);
        var root   = telemetry.RootElement;
        var cartId = root.GetProperty("cart_id").GetString() ?? "unknown";
        var aisle  = root.TryGetProperty("active_aisle", out var a)
                         ? a.GetString() ?? "NONE" : "NONE";
        var tsMs   = root.GetProperty("timestamp_ms").GetInt64();

        // Write to Firestore with 72-hour TTL via scheduled Cloud Function deletion
        var docRef = _db
            .Collection("CartJourneys")
            .Document($"{cartId}_{tsMs}");

        await docRef.SetAsync(new
        {
            cart_id   = cartId,
            aisle     = aisle,
            timestamp = Timestamp.FromDateTimeOffset(
                            DateTimeOffset.FromUnixTimeMilliseconds(tsMs)),
            ttl_epoch = DateTimeOffset.UtcNow.AddHours(72).ToUnixTimeSeconds()
        });

        _log.LogInformation(
            "GCP: Journey record written — Cart={CartId} Aisle={Aisle}", cartId, aisle);

        ctx.Response.StatusCode = StatusCodes.Status204NoContent;
    }
}
