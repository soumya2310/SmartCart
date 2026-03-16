// =============================================================================
// Supplementary Material S1.5 (Azure Functions portion)
// CloudFunctionsAll.cs
// Cloud-Integrated AI-Driven Motorized Shopping Cart — Azure Functions
//
// Author : soumya2310 (Soumya Chattopadhyay)
// Affil  : Biju Patnaik University of Technology, Odisha, India
// Email  : soumya2310@gmail.com
// Paper  : "Edge-Cloud Collaborative Intelligence for Autonomous IoT Navigation"
//           Journal of Cloud Computing (Springer), 2025
//
// Azure Functions isolated worker model (C# / .NET 8):
//   - CartAnalyticsFunction     : triggered by Azure IoT Hub event
//   - InventoryOptimizerFunction: timer-triggered every 15 minutes
//   - AlertDispatcherFunction   : HTTP trigger for store-ops mobile push
//
// Target:  .NET 8 LTS  |  C# 12
// NuGet:   Microsoft.Azure.Functions.Worker (1.22.x)
//          Microsoft.Azure.Functions.Worker.Extensions.EventHubs (6.x)
//          Microsoft.Azure.Functions.Worker.Extensions.Timer (4.x)
//          Microsoft.Azure.Functions.Worker.Extensions.Http (3.x)
//          Azure.Data.Tables (12.x)
//
// NAMESPACE FIX
// ─────────────────────────────────────────────────────────────────────────────
// The original file contained TWO file-scoped namespace declarations:
//
//     namespace SmartCart.Azure.Functions;   // ← first declaration
//     ...
//     namespace SmartCart.GCP.Functions;     // ← second declaration (ERROR)
//
// C# only allows ONE file-scoped namespace per file (CS8955). Having two
// caused the compiler to reject the entire file, which is why AuthorizationLevel
// and other well-formed symbols appeared as unresolved (CS0103).
//
// Fix applied:
//   • This file now contains ONLY the Azure Functions namespace.
//   • The GCP Functions classes have been moved to GcpFunctionsAll.cs.
//
// DUPLICATE TYPE FIX
// ─────────────────────────────────────────────────────────────────────────────
// The original CloudFunctionsAll.cs redefined CartTelemetryEvent in namespace
// SmartCart.Azure.Functions, which conflicts with the identical-namespace
// definition in AzureFunctionsTelemetryProcessor.cs (different fields → CS0101).
// Fix: the analytics event payload in this file is renamed CartAnalyticsEvent
// to reflect its distinct field shape (BatterySoC, NavigationGoal, SensorsOk).
// =============================================================================

using Microsoft.Azure.Functions.Worker;         // ← AuthorizationLevel is here (fixes CS0103)
using Microsoft.Azure.Functions.Worker.Http;
using Microsoft.Extensions.Logging;
using Azure.Data.Tables;
using System.Net;
using System.Text.Json;
using System.Text.Json.Serialization;

// ONE file-scoped namespace — the GCP classes are in GcpFunctionsAll.cs
namespace SmartCart.Azure.Functions;

// ---------------------------------------------------------------------------
// Shared models (Azure analytics functions)
// ---------------------------------------------------------------------------

/// <summary>
/// Cart telemetry event for the analytics and inventory functions.
/// NOTE: this has a different field set from the streaming telemetry event in
/// AzureFunctionsTelemetryProcessor.cs and is intentionally named differently
/// to avoid the CS0101 duplicate-type error.
/// </summary>
public record CartAnalyticsEvent(
    string CartId,
    double X,
    double Y,
    double BatterySoC,
    string NavigationGoal,
    bool   SensorsOk);

public record ShelfRecommendation(
    string Aisle,
    string ProductId,
    int    SuggestedFacings,
    double DwellScore);

[JsonSerializable(typeof(CartAnalyticsEvent))]
[JsonSerializable(typeof(ShelfRecommendation))]
[JsonSerializable(typeof(List<ShelfRecommendation>))]
internal partial class AzureJsonContext : JsonSerializerContext { }

// ---------------------------------------------------------------------------
// Cart analytics — triggered on each IoT Hub message
// ---------------------------------------------------------------------------
public class CartAnalyticsFunction(
    TableServiceClient tableService,
    ILogger<CartAnalyticsFunction> log)
{
    private readonly TableClient _journeys =
        tableService.GetTableClient("CartJourneys");

    [Function("CartAnalytics")]
    public async Task RunAsync(
        [EventHubTrigger("cart-telemetry", Connection = "IoTHubConnection")]
        string[] messages,
        FunctionContext ctx)
    {
        foreach (var msg in messages)
        {
            CartAnalyticsEvent? evt;
            try
            {
                evt = JsonSerializer.Deserialize(msg,
                          AzureJsonContext.Default.CartAnalyticsEvent);
            }
            catch (JsonException ex)
            {
                log.LogWarning(ex, "Failed to deserialise telemetry message");
                continue;
            }
            if (evt is null) continue;

            var entity = new TableEntity(evt.CartId, DateTime.UtcNow.Ticks.ToString())
            {
                ["CartId"]     = evt.CartId,
                ["X"]          = evt.X,
                ["Y"]          = evt.Y,
                ["BatterySoC"] = evt.BatterySoC,
                ["NavGoal"]    = evt.NavigationGoal,
                ["SensorsOk"]  = evt.SensorsOk,
                ["Timestamp"]  = DateTimeOffset.UtcNow
            };

            await _journeys.UpsertEntityAsync(entity);
            log.LogDebug("Journey record written: Cart={CartId}", evt.CartId);
        }
    }
}

// ---------------------------------------------------------------------------
// Inventory optimiser — timer trigger every 15 minutes during store hours
// ---------------------------------------------------------------------------
public class InventoryOptimizerFunction(
    TableServiceClient tableService,
    ILogger<InventoryOptimizerFunction> log)
{
    [Function("InventoryOptimizer")]
    public async Task RunAsync(
        [TimerTrigger("0 */15 8-22 * * *")] TimerInfo timer,
        FunctionContext ctx)
    {
        log.LogInformation("InventoryOptimizer triggered at {Now}", DateTime.UtcNow);

        var dwellScores     = await ComputeAisleDwellScoresAsync();
        var recommendations = GenerateShelfRecommendations(dwellScores);

        log.LogInformation(
            "Generated {Count} shelf-placement recommendations", recommendations.Count);

        // Write recommendations to Table Storage for store-ops dashboard
        var recTable = tableService.GetTableClient("ShelfRecommendations");
        foreach (var rec in recommendations)
        {
            var entity = new TableEntity("current", rec.Aisle)
            {
                ["ProductId"]        = rec.ProductId,
                ["SuggestedFacings"] = rec.SuggestedFacings,
                ["DwellScore"]       = rec.DwellScore,
                ["GeneratedAt"]      = DateTimeOffset.UtcNow
            };
            await recTable.UpsertEntityAsync(entity);
        }

        if (timer.IsPastDue)
            log.LogWarning("InventoryOptimizer timer ran past its scheduled time.");
    }

    private static Task<Dictionary<string, double>> ComputeAisleDwellScoresAsync()
    {
        // Production: aggregate CartJourneys over trailing 24h window.
        // Placeholder returns synthetic scores for demonstration.
        return Task.FromResult(new Dictionary<string, double>
        {
            ["A1"] = 0.87, ["A2"] = 0.43, ["A3"] = 0.91,
            ["B1"] = 0.65, ["B2"] = 0.78, ["C1"] = 0.34
        });
    }

    private static List<ShelfRecommendation> GenerateShelfRecommendations(
        Dictionary<string, double> dwellScores)
    {
        return dwellScores
            .Where(kv => kv.Value > 0.70)
            .Select(kv => new ShelfRecommendation(
                Aisle:            kv.Key,
                ProductId:        $"PROMO_{kv.Key}",
                SuggestedFacings: (int)(kv.Value * 6),
                DwellScore:       kv.Value))
            .ToList();
    }
}

// ---------------------------------------------------------------------------
// Alert dispatcher — HTTP trigger for maintenance push notifications
// ---------------------------------------------------------------------------
public class AlertDispatcherFunction(ILogger<AlertDispatcherFunction> log)
{
    public record AlertRequest(string CartId, string AlertType, string Detail);

    [Function("AlertDispatcher")]
    public async Task<HttpResponseData> RunAsync(
        // FIX CS0103: AuthorizationLevel lives in Microsoft.Azure.Functions.Worker.
        // Resolved by the 'using Microsoft.Azure.Functions.Worker;' at the top
        // of this file (previously the second namespace declaration shadowed it).
        [HttpTrigger(AuthorizationLevel.Function, "post", Route = "alerts")]
        HttpRequestData req,
        FunctionContext ctx)
    {
        AlertRequest? alert;
        try
        {
            alert = await JsonSerializer.DeserializeAsync<AlertRequest>(req.Body);
        }
        catch
        {
            var bad = req.CreateResponse(HttpStatusCode.BadRequest);
            await bad.WriteStringAsync("Invalid alert payload");
            return bad;
        }

        log.LogWarning(
            "Cart alert: CartId={CartId} Type={Type} Detail={Detail}",
            alert?.CartId, alert?.AlertType, alert?.Detail);

        // Production: send push via Azure Notification Hubs to store-ops mobile app
        var response = req.CreateResponse(HttpStatusCode.Accepted);
        await response.WriteStringAsync("Alert dispatched");
        return response;
    }
}
