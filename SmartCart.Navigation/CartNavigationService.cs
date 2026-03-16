// =============================================================================
// Author : soumya2310 (Soumya Chattopadhyay)
// Affil  : Biju Patnaik University of Technology, Odisha, India
// Email  : soumya2310@gmail.com
// Paper  : "Edge-Cloud Collaborative Intelligence for Autonomous IoT Navigation"
//           Journal of Cloud Computing (Springer), 2025
// Supplementary Material S1.1
// CartNavigationService.cs
// Cloud-Integrated AI-Driven Motorized Shopping Cart — Navigation Layer
//
// Resolves product names to store-coordinate waypoints via Amazon DynamoDB
// and dispatches navigation goals to the ROS 2 Nav2 stack through a named pipe.
//
// Target:  .NET 8 LTS  |  C# 12
// NuGet:   AWSSDK.DynamoDBv2 (3.7.x)
//          Microsoft.Extensions.Logging.Abstractions (8.x)
//          Microsoft.Extensions.DependencyInjection.Abstractions (8.x)
// =============================================================================

using Amazon.DynamoDBv2;
using Amazon.DynamoDBv2.Model;
using Microsoft.Extensions.Logging;
using System.IO.Pipes;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace SmartCart.Navigation;

// ---------------------------------------------------------------------------
// Domain model — immutable record
// ---------------------------------------------------------------------------
public record ProductLocation(
    string ProductId,
    string ProductName,
    string Aisle,
    double X,
    double Y,
    double Orientation);

// ---------------------------------------------------------------------------
// Navigation goal message sent to ROS 2 Nav2 via named pipe
// ---------------------------------------------------------------------------
[JsonSerializable(typeof(Nav2GoalMessage))]
internal partial class Nav2JsonContext : JsonSerializerContext { }

internal record Nav2GoalMessage(
    double X, double Y, double Orientation,
    string Aisle, string ProductName,
    long IssuedAtUtcMs);

// ---------------------------------------------------------------------------
// Service interface (for DI / unit testing)
// ---------------------------------------------------------------------------
public interface ICartNavigationService
{
    Task<ProductLocation?> LookupProductAsync(string productName,
                                              CancellationToken ct = default);

    Task<NavigationResult> NavigateToProductAsync(ProductLocation location,
                                                  CancellationToken ct = default);
}

public enum NavigationResult { Dispatched, ProductNotFound, PipeTimeout, Error }

// ---------------------------------------------------------------------------
// Production implementation
// ---------------------------------------------------------------------------
public sealed class CartNavigationService : ICartNavigationService
{
    private const string TableName     = "StoreProducts";
    private const string PipeName      = "nav2-goal";
    private const int    PipeTimeoutMs = 2_000;

    private readonly IAmazonDynamoDB                   _dynamo;
    private readonly ILogger<CartNavigationService>    _log;

    public CartNavigationService(
        IAmazonDynamoDB dynamo,
        ILogger<CartNavigationService> log)
    {
        _dynamo = dynamo;
        _log    = log;
    }

    /// <summary>
    /// Resolves a product name to its store-coordinate location.
    /// Returns null when the product is not present in the DynamoDB catalog.
    /// </summary>
    public async Task<ProductLocation?> LookupProductAsync(
        string productName, CancellationToken ct = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(productName);

        var request = new GetItemRequest
        {
            TableName = TableName,
            Key       = new() { ["product_name"] = new AttributeValue(productName) },
            ConsistentRead = false   // eventual-consistent for sub-ms latency
        };

        GetItemResponse response;
        try
        {
            response = await _dynamo.GetItemAsync(request, ct);
        }
        catch (Exception ex)
        {
            _log.LogError(ex, "DynamoDB lookup failed for product: {Name}", productName);
            throw;
        }

        if (!response.IsItemSet)
        {
            _log.LogWarning("Product not found in catalog: {Name}", productName);
            return null;
        }

        var item = response.Item;
        return new ProductLocation(
            ProductId:   item["product_id"].S,
            ProductName: item["product_name"].S,
            Aisle:       item["aisle"].S,
            X:           double.Parse(item["x"].N),
            Y:           double.Parse(item["y"].N),
            Orientation: double.Parse(item["orientation"].N));
    }

    /// <summary>
    /// Dispatches a navigation goal to the ROS 2 Nav2 BasicNavigator via named pipe.
    /// The Nav2 bridge process reads JSON goal messages and calls goToPose().
    /// </summary>
    public async Task<NavigationResult> NavigateToProductAsync(
        ProductLocation location, CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(location);

        var goal = new Nav2GoalMessage(
            X:             location.X,
            Y:             location.Y,
            Orientation:   location.Orientation,
            Aisle:         location.Aisle,
            ProductName:   location.ProductName,
            IssuedAtUtcMs: DateTimeOffset.UtcNow.ToUnixTimeMilliseconds());

        var json = JsonSerializer.Serialize(goal, Nav2JsonContext.Default.Nav2GoalMessage);

        try
        {
            await using var pipe = new NamedPipeClientStream(
                ".", PipeName, PipeDirection.Out, PipeOptions.Asynchronous);

            await pipe.ConnectAsync(PipeTimeoutMs, ct);

            await using var writer = new StreamWriter(pipe, leaveOpen: true);
            await writer.WriteLineAsync(json.AsMemory(), ct);
            await writer.FlushAsync(ct);

            _log.LogInformation(
                "Navigation goal dispatched → Aisle {Aisle} ({Name}) at [{X:F2}, {Y:F2}]",
                location.Aisle, location.ProductName, location.X, location.Y);

            return NavigationResult.Dispatched;
        }
        catch (TimeoutException)
        {
            _log.LogError("Named pipe connect timeout after {Ms} ms — Nav2 bridge not running?",
                PipeTimeoutMs);
            return NavigationResult.PipeTimeout;
        }
        catch (Exception ex)
        {
            _log.LogError(ex, "Navigation goal dispatch failed");
            return NavigationResult.Error;
        }
    }
}
