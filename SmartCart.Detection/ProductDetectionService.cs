// =============================================================================
// Author : soumya2310 (Soumya Chattopadhyay)
// Affil  : Biju Patnaik University of Technology, Odisha, India
// Email  : soumya2310@gmail.com
// Paper  : "Edge-Cloud Collaborative Intelligence for Autonomous IoT Navigation"
//           Journal of Cloud Computing (Springer), 2025
// Supplementary Material S1.4
// ProductDetectionService.cs
// Cloud-Integrated AI-Driven Motorized Shopping Cart — Edge Detection Layer
//
// Runs YOLOv8-L exported to ONNX on the NVIDIA Jetson AGX Orin using
// Microsoft.ML.OnnxRuntime with CUDA execution provider.
// Fuses RGB and aligned depth frames from Intel RealSense D455 to produce
// metric-space DetectedProduct records with sub-15 ms per-frame latency.
//
// Target:  .NET 8 LTS  |  C# 12  (Linux/ARM64 on Jetson)
// NuGet:   Microsoft.ML.OnnxRuntime.Gpu (1.18.x)
//          librealsense2 P/Invoke bindings (community: LibRealSense.Net)
// =============================================================================

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.Extensions.Logging;
using System.Runtime.InteropServices;

namespace SmartCart.Detection;

// ---------------------------------------------------------------------------
// Domain models
// ---------------------------------------------------------------------------
public record BoundingBox(int X1, int Y1, int X2, int Y2)
{
    public int CenterX => (X1 + X2) / 2;
    public int CenterY => (Y1 + Y2) / 2;
    public int Width   => X2 - X1;
    public int Height  => Y2 - Y1;
}

public record DetectedProduct(
    string  ClassName,
    float   Confidence,
    BoundingBox BBox,
    float   DepthMeters);          // 0 = depth unavailable / filtered

// ---------------------------------------------------------------------------
// YOLOv8 ONNX inference configuration
// ---------------------------------------------------------------------------
public sealed class DetectionConfig
{
    public string ModelPath      { get; init; } = "models/retail_yolov8l.onnx";
    public int    InputWidth     { get; init; } = 640;
    public int    InputHeight    { get; init; } = 640;
    public float  ConfThreshold  { get; init; } = 0.65f;
    public float  IouThreshold   { get; init; } = 0.45f;
    public bool   UseCuda        { get; init; } = true;
    public int    GpuDeviceId    { get; init; } = 0;
}

// ---------------------------------------------------------------------------
// Production detection service
// ---------------------------------------------------------------------------
public sealed class ProductDetectionService : IDisposable
{
    private readonly InferenceSession              _session;
    private readonly DetectionConfig               _cfg;
    private readonly ILogger<ProductDetectionService> _log;
    private readonly string[]                      _classNames;

    // Input tensor reused across frames to avoid allocations
    private readonly float[] _inputBuffer;

    public ProductDetectionService(
        DetectionConfig cfg,
        string[]        classNames,
        ILogger<ProductDetectionService> log)
    {
        _cfg        = cfg;
        _classNames = classNames;
        _log        = log;
        _inputBuffer = new float[3 * cfg.InputWidth * cfg.InputHeight];

        var sessionOpts = new SessionOptions();
        if (cfg.UseCuda)
        {
            sessionOpts.AppendExecutionProvider_CUDA(cfg.GpuDeviceId);
            _log.LogInformation(
                "ONNX Runtime: CUDA EP on GPU {Id}", cfg.GpuDeviceId);
        }
        sessionOpts.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

        _session = new InferenceSession(cfg.ModelPath, sessionOpts);
        _log.LogInformation(
            "YOLOv8-L ONNX model loaded from {Path}", cfg.ModelPath);
    }

    /// <summary>
    /// Runs object detection on a pre-decoded RGB frame (HWC, uint8).
    /// depthFrame is a float[H,W] metric depth map; pass null to skip depth fusion.
    /// </summary>
    public IReadOnlyList<DetectedProduct> Detect(
        byte[,,] rgbFrame,     // [H, W, 3]
        float[,]? depthFrame)  // [H, W] metres  (nullable)
    {
        int srcH = rgbFrame.GetLength(0);
        int srcW = rgbFrame.GetLength(1);

        PreprocessFrame(rgbFrame, srcH, srcW);

        var inputTensor = new DenseTensor<float>(
            _inputBuffer,
            new[] { 1, 3, _cfg.InputHeight, _cfg.InputWidth });

        var inputs = new NamedOnnxValue[]
        {
            NamedOnnxValue.CreateFromTensor("images", inputTensor)
        };

        using var outputs = _session.Run(inputs);
        var rawOutput = outputs.First().AsTensor<float>();

        return PostProcess(rawOutput, srcW, srcH, depthFrame);
    }

    // ── Letterbox resize + normalise to [0,1] ────────────────────────────────
    private void PreprocessFrame(byte[,,] src, int srcH, int srcW)
    {
        float scaleX = (float)_cfg.InputWidth  / srcW;
        float scaleY = (float)_cfg.InputHeight / srcH;
        float scale  = Math.Min(scaleX, scaleY);
        int   newW   = (int)(srcW * scale);
        int   newH   = (int)(srcH * scale);
        int   padX   = (_cfg.InputWidth  - newW) / 2;
        int   padY   = (_cfg.InputHeight - newH) / 2;

        Array.Clear(_inputBuffer, 0, _inputBuffer.Length);

        int planeSize = _cfg.InputWidth * _cfg.InputHeight;

        for (int dy = 0; dy < newH; dy++)
        for (int dx = 0; dx < newW; dx++)
        {
            int sx = (int)(dx / scale);
            int sy = (int)(dy / scale);
            sx = Math.Clamp(sx, 0, srcW - 1);
            sy = Math.Clamp(sy, 0, srcH - 1);

            int outX = dx + padX;
            int outY = dy + padY;
            int idx  = outY * _cfg.InputWidth + outX;

            _inputBuffer[idx]                = src[sy, sx, 0] / 255f; // R
            _inputBuffer[idx + planeSize]    = src[sy, sx, 1] / 255f; // G
            _inputBuffer[idx + planeSize*2]  = src[sy, sx, 2] / 255f; // B
        }
    }

    // ── Decode YOLOv8 output [1, 84, 8400] with NMS ──────────────────────────
    private IReadOnlyList<DetectedProduct> PostProcess(
        Tensor<float> output, int srcW, int srcH, float[,]? depth)
    {
        int numClasses  = output.Dimensions[1] - 4;  // 80 for COCO-based models
        int numAnchors  = output.Dimensions[2];       // 8400

        var candidates = new List<(BoundingBox Box, float Conf, int Class)>();

        for (int a = 0; a < numAnchors; a++)
        {
            // Find max class score
            int   bestClass = 0;
            float bestScore = 0f;
            for (int c = 0; c < numClasses; c++)
            {
                float s = output[0, 4 + c, a];
                if (s > bestScore) { bestScore = s; bestClass = c; }
            }
            if (bestScore < _cfg.ConfThreshold) continue;

            // cx, cy, w, h in model input space
            float cx = output[0, 0, a];
            float cy = output[0, 1, a];
            float bw = output[0, 2, a];
            float bh = output[0, 3, a];

            // Scale to source image coordinates
            float sx = (float)srcW / _cfg.InputWidth;
            float sy = (float)srcH / _cfg.InputHeight;

            int x1 = (int)((cx - bw / 2) * sx);
            int y1 = (int)((cy - bh / 2) * sy);
            int x2 = (int)((cx + bw / 2) * sx);
            int y2 = (int)((cy + bh / 2) * sy);

            x1 = Math.Clamp(x1, 0, srcW - 1);
            y1 = Math.Clamp(y1, 0, srcH - 1);
            x2 = Math.Clamp(x2, 0, srcW - 1);
            y2 = Math.Clamp(y2, 0, srcH - 1);

            candidates.Add((new BoundingBox(x1, y1, x2, y2), bestScore, bestClass));
        }

        // Non-maximum suppression per class
        var results = NMS(candidates);

        return results.Select(r =>
        {
            float d = 0f;
            if (depth is not null)
            {
                int cx = r.Box.CenterX;
                int cy = r.Box.CenterY;
                if (cx < depth.GetLength(1) && cy < depth.GetLength(0))
                    d = depth[cy, cx];
            }

            string label = r.Class < _classNames.Length
                               ? _classNames[r.Class]
                               : $"class_{r.Class}";

            return new DetectedProduct(label, r.Conf, r.Box, d);
        }).ToList();
    }

    // Greedy NMS
    private static List<(BoundingBox Box, float Conf, int Class)> NMS(
        List<(BoundingBox Box, float Conf, int Class)> boxes)
    {
        var sorted  = boxes.OrderByDescending(b => b.Conf).ToList();
        var keep    = new List<(BoundingBox Box, float Conf, int Class)>();

        while (sorted.Count > 0)
        {
            var best = sorted[0];
            keep.Add(best);
            sorted.RemoveAt(0);
            sorted.RemoveAll(b =>
                b.Class == best.Class && IoU(best.Box, b.Box) > 0.45f);
        }
        return keep;
    }

    private static float IoU(BoundingBox a, BoundingBox b)
    {
        int interX1 = Math.Max(a.X1, b.X1);
        int interY1 = Math.Max(a.Y1, b.Y1);
        int interX2 = Math.Min(a.X2, b.X2);
        int interY2 = Math.Min(a.Y2, b.Y2);
        float inter = Math.Max(0, interX2 - interX1) * Math.Max(0, interY2 - interY1);
        float aArea = a.Width * a.Height;
        float bArea = b.Width * b.Height;
        return inter / (aArea + bArea - inter + 1e-6f);
    }

    public void Dispose() => _session.Dispose();
}
