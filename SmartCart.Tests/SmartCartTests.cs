// =============================================================================
// Author : soumya2310 (Soumya Chattopadhyay)
// Affil  : Biju Patnaik University of Technology, Odisha, India
// Email  : soumya2310@gmail.com
// Paper  : "Edge-Cloud Collaborative Intelligence for Autonomous IoT Navigation"
//           Journal of Cloud Computing (Springer), 2025
// Supplementary Material S1.6
// SmartCart.Tests — xUnit Unit Tests for Navigation and Telemetry Services
//
// Demonstrates testability of the DI-injected C# services using mock objects.
// All tests run without real AWS/Azure credentials (fully offline).
//
// Target:  .NET 8 LTS  |  C# 12
// NuGet:   xunit (2.9.x)
//          Moq (4.20.x)
//          Microsoft.Extensions.Logging.Abstractions (8.x)
//          AWSSDK.DynamoDBv2 (3.7.x)
// =============================================================================

using Amazon.DynamoDBv2;
using Amazon.DynamoDBv2.Model;
using Microsoft.Extensions.Logging.Abstractions;
using Moq;
using SmartCart.Navigation;
using SmartCart.Detection;
using Xunit;

namespace SmartCart.Tests;

// ---------------------------------------------------------------------------
// CartNavigationService tests
// ---------------------------------------------------------------------------
public class CartNavigationServiceTests
{
    private readonly Mock<IAmazonDynamoDB>    _dynamoMock = new();
    private readonly CartNavigationService    _sut;

    public CartNavigationServiceTests()
    {
        _sut = new CartNavigationService(
            _dynamoMock.Object,
            NullLogger<CartNavigationService>.Instance);
    }

    [Fact]
    public async Task LookupProduct_KnownProduct_ReturnsCorrectLocation()
    {
        // Arrange
        var fakeItem = new Dictionary<string, AttributeValue>
        {
            ["product_id"]   = new("P001"),
            ["product_name"] = new("Whole Milk"),
            ["aisle"]        = new("Dairy-B3"),
            ["x"]            = new() { N = "12.50" },
            ["y"]            = new() { N = "4.75"  },
            ["orientation"]  = new() { N = "0.0"   }
        };

        _dynamoMock
            .Setup(d => d.GetItemAsync(
                It.Is<GetItemRequest>(r => r.TableName == "StoreProducts"),
                It.IsAny<CancellationToken>()))
            .ReturnsAsync(new GetItemResponse { Item = fakeItem });

        // Act
        var result = await _sut.LookupProductAsync("Whole Milk");

        // Assert
        Assert.NotNull(result);
        Assert.Equal("Dairy-B3", result.Aisle);
        Assert.Equal(12.50, result.X, precision: 2);
        Assert.Equal(4.75,  result.Y, precision: 2);
    }

    [Fact]
    public async Task LookupProduct_UnknownProduct_ReturnsNull()
    {
        // Arrange: DynamoDB returns response with no Item
        _dynamoMock
            .Setup(d => d.GetItemAsync(
                It.IsAny<GetItemRequest>(),
                It.IsAny<CancellationToken>()))
            .ReturnsAsync(new GetItemResponse { Item = [] });

        // Act
        var result = await _sut.LookupProductAsync("NonExistentBrand");

        // Assert
        Assert.Null(result);
    }

    [Fact]
    public async Task LookupProduct_NullOrEmpty_ThrowsArgumentException()
    {
        await Assert.ThrowsAsync<ArgumentException>(
            () => _sut.LookupProductAsync(string.Empty));

        await Assert.ThrowsAsync<ArgumentException>(
            () => _sut.LookupProductAsync("   "));
    }

    [Fact]
    public async Task LookupProduct_DynamoException_Propagates()
    {
        _dynamoMock
            .Setup(d => d.GetItemAsync(
                It.IsAny<GetItemRequest>(),
                It.IsAny<CancellationToken>()))
            .ThrowsAsync(new AmazonDynamoDBException("Connection refused"));

        await Assert.ThrowsAsync<AmazonDynamoDBException>(
            () => _sut.LookupProductAsync("Whole Milk"));
    }
}

// ---------------------------------------------------------------------------
// ProductDetectionService tests
// ---------------------------------------------------------------------------
public class ProductDetectionServiceTests
{
    [Fact]
    public void BoundingBox_CenterCalculation_IsCorrect()
    {
        var box = new BoundingBox(100, 200, 300, 400);
        Assert.Equal(200, box.CenterX);
        Assert.Equal(300, box.CenterY);
        Assert.Equal(200, box.Width);
        Assert.Equal(200, box.Height);
    }

    [Fact]
    public void DetectedProduct_DepthFilterThreshold_FiltersInvalidReturns()
    {
        // Depth values below 0.1 m are filtered as invalid (RealSense min range)
        var validProduct   = new DetectedProduct("Milk", 0.92f, new(10,10,50,50), 1.2f);
        var invalidProduct = new DetectedProduct("Milk", 0.92f, new(10,10,50,50), 0.05f);

        Assert.True(validProduct.DepthMeters  >= 0.1f);
        Assert.False(invalidProduct.DepthMeters >= 0.1f);
    }

    [Fact]
    public void DetectedProduct_ConfidenceAboveThreshold_PassesFilter()
    {
        const float threshold = 0.65f;
        var products = new[]
        {
            new DetectedProduct("A", 0.90f, new(0,0,10,10), 1.5f),
            new DetectedProduct("B", 0.45f, new(0,0,10,10), 2.0f),
            new DetectedProduct("C", 0.65f, new(0,0,10,10), 0.8f),
        };

        var passing = products.Where(p => p.Confidence >= threshold).ToList();
        Assert.Equal(2, passing.Count);
        Assert.All(passing, p => Assert.True(p.Confidence >= threshold));
    }
}

// ---------------------------------------------------------------------------
// ProductLocation record tests
// ---------------------------------------------------------------------------
public class ProductLocationTests
{
    [Fact]
    public void ProductLocation_EqualityByValue()
    {
        var a = new ProductLocation("P1","Milk","Dairy",1.0,2.0,0.0);
        var b = new ProductLocation("P1","Milk","Dairy",1.0,2.0,0.0);
        Assert.Equal(a, b);
    }

    [Fact]
    public void ProductLocation_WhenXDiffers_NotEqual()
    {
        var a = new ProductLocation("P1","Milk","Dairy",1.0,2.0,0.0);
        var b = new ProductLocation("P1","Milk","Dairy",9.9,2.0,0.0);
        Assert.NotEqual(a, b);
    }
}
