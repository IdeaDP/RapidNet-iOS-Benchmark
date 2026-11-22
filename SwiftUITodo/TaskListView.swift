import CoreML
import Foundation
import SwiftUI

/// Deep Focus home screen. (Type name kept as `TaskListView` because SceneDelegate creates it.)
struct TaskListView: View {
    @State private var cameraMode: CameraMode?

    enum CameraMode: String, Identifiable {
        case segmentation, detection
        var id: String { rawValue }
    }

    var body: some View {
        NavigationStack {
            ZStack {
                DFBackground()
                ScrollView {
                    VStack(alignment: .leading, spacing: 20) {
                        header
                        LazyVGrid(columns: [GridItem(.flexible(), spacing: 14), GridItem(.flexible(), spacing: 14)], spacing: 14) {
                            NavigationLink { ClassificationView() } label: {
                                FeatureTile(title: "Classify", subtitle: "Photos · 1000 classes · 3 levels",
                                            icon: "photo.on.rectangle.angled", tint: DF.sky)
                            }
                            NavigationLink { BenchmarkView() } label: {
                                FeatureTile(title: "Benchmark", subtitle: "Neural Engine latency",
                                            icon: "speedometer", tint: Color(red: 0.62, green: 0.55, blue: 1.0))
                            }
                            Button { cameraMode = .segmentation } label: {
                                FeatureTile(title: "Segmentation", subtitle: "Live camera · ADE20K",
                                            icon: "square.stack.3d.down.forward.fill", tint: Color(red: 0.42, green: 0.85, blue: 0.45))
                            }
                            Button { cameraMode = .detection } label: {
                                FeatureTile(title: "Detection", subtitle: "Live camera · COCO",
                                            icon: "viewfinder", tint: Color(red: 0.98, green: 0.62, blue: 0.27))
                            }
                        }
                        .buttonStyle(.plain)
                        aboutCard
                    }
                    .padding()
                }
            }
            .navigationBarHidden(true)
        }
        .preferredColorScheme(.dark)
        .tint(DF.sky)
        .fullScreenCover(item: $cameraMode) { mode in
            NavigationStack {
                SegmentationCamera(detectsObjects: mode == .detection)
                    .navigationTitle(mode == .detection ? "Object detection" : "Semantic segmentation")
                    .navigationBarTitleDisplayMode(.inline)
                    .toolbar {
                        ToolbarItem(placement: .cancellationAction) {
                            Button("Done") { cameraMode = nil }
                        }
                    }
            }
            .preferredColorScheme(.dark)
        }
    }

    private var header: some View {
        HStack(spacing: 16) {
            DFLogo(size: 64)
            VStack(alignment: .leading, spacing: 4) {
                Text("Deep Focus")
                    .font(.system(size: 32, weight: .heavy, design: .rounded))
                    .foregroundStyle(Color.white)
                Text("RapidNet on iPhone · EN3160")
                    .font(.subheadline).foregroundStyle(DF.cyan)
            }
        }
        .padding(.top, 12)
    }

    private var aboutCard: some View {
        DFCard {
            DFSectionTitle(title: "About RapidNet", icon: "info.circle.fill")
            Text("A pure-CNN mobile backbone built on multi-level dilated convolutions (WACV 2025). Every model in this app runs on-device with Core ML.")
                .font(.subheadline).foregroundStyle(Color.white.opacity(0.75))
            HStack(spacing: 8) {
                ForEach(["Ti", "S", "M", "B"], id: \.self) { v in
                    Text("RapidNet-\(v)")
                        .font(.caption.weight(.semibold))
                        .padding(.horizontal, 10).padding(.vertical, 6)
                        .foregroundStyle(Color.white.opacity(0.85))
                        .background(Color.white.opacity(0.08), in: Capsule())
                }
            }
        }
    }
}

struct FeatureTile: View {
    let title: String
    let subtitle: String
    let icon: String
    let tint: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Image(systemName: icon)
                .font(.system(size: 22, weight: .semibold))
                .foregroundStyle(DF.navyBottom)
                .frame(width: 46, height: 46)
                .background(tint, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            Spacer(minLength: 0)
            Text(title).font(.headline).foregroundStyle(Color.white)
            Text(subtitle).font(.caption).foregroundStyle(Color.white.opacity(0.6)).lineLimit(2)
        }
        .padding(16)
        .frame(maxWidth: .infinity, minHeight: 160, alignment: .leading)
        .background(
            LinearGradient(colors: [tint.opacity(0.22), Color.white.opacity(0.05)], startPoint: .topLeading, endPoint: .bottomTrailing),
            in: RoundedRectangle(cornerRadius: 22, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 22, style: .continuous).stroke(Color.white.opacity(0.10), lineWidth: 1))
    }
}

// MARK: - Benchmark (same measurement as before: 50 rounds × 50 predictions, trim 10 at each end)

struct BenchmarkResult {
    let variant: String
    let meanMs: Double
    let retained: Int
    let thermalStart: String
    let thermalEnd: String
}

struct BenchmarkView: View {
    private let variants = ["Ti", "S", "M", "B"].filter {
        Bundle.main.url(forResource: "RapidNet_\($0)", withExtension: "mlmodelc") != nil
    }
    @State private var selection = "M"
    @State private var isRunning = false
    @State private var results: [String: BenchmarkResult] = [:]
    @State private var errorText: String?

    var body: some View {
        ZStack {
            DFBackground()
            ScrollView {
                VStack(spacing: 16) {
                    DFCard {
                        DFSectionTitle(title: "Model", icon: "cpu")
                        HStack(spacing: 8) {
                            ForEach(variants, id: \.self) { v in
                                DFChip(title: "RapidNet-\(v)", selected: v == selection) { selection = v }
                            }
                        }
                        .disabled(isRunning)
                        Text("50 rounds · 50 predictions per round · trim 10 at each end · CPU + Neural Engine · 224×224")
                            .font(.caption).foregroundStyle(Color.white.opacity(0.6))
                        Button(action: runBenchmark) {
                            HStack {
                                if isRunning { ProgressView().tint(DF.navyBottom) }
                                Text(isRunning ? "Running RapidNet-\(selection)…" : "Run Benchmark").font(.headline)
                            }
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 14)
                            .foregroundStyle(DF.navyBottom)
                            .background(DF.accent, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                        }
                        .buttonStyle(.plain)
                        .disabled(isRunning || variants.isEmpty)
                    }

                    if let errorText {
                        DFCard { Label(errorText, systemImage: "exclamationmark.triangle.fill").foregroundStyle(Color.orange) }
                    }

                    if let r = results[selection] {
                        DFCard {
                            DFSectionTitle(title: "RapidNet-\(r.variant)", icon: "gauge.with.dots.needle.67percent")
                            HStack(alignment: .firstTextBaseline, spacing: 6) {
                                Text(String(format: "%.3f", r.meanMs))
                                    .font(.system(size: 48, weight: .bold, design: .rounded).monospacedDigit())
                                    .foregroundStyle(Color.white)
                                Text("ms / prediction").foregroundStyle(Color.white.opacity(0.6))
                            }
                            HStack(spacing: 10) {
                                stat("Throughput", String(format: "%.1f /s", 1000 / r.meanMs))
                                stat("Samples", "\(r.retained)")
                                stat("Thermal", "\(r.thermalStart) → \(r.thermalEnd)")
                            }
                        }
                        .textSelection(.enabled)
                    }

                    if results.count > 1 {
                        DFCard {
                            DFSectionTitle(title: "Compared", icon: "chart.bar.fill")
                            let maxMs = results.values.map(\.meanMs).max() ?? 1
                            ForEach(variants.filter { results[$0] != nil }, id: \.self) { v in
                                let r = results[v]!
                                ProbabilityBarLatency(label: "RapidNet-\(v)", ms: r.meanMs, fraction: r.meanMs / maxMs)
                            }
                        }
                    }
                }
                .padding()
            }
        }
        .navigationTitle("Benchmark")
        .navigationBarTitleDisplayMode(.inline)
        .toolbarColorScheme(.dark, for: .navigationBar)
        .onAppear {
            if !variants.contains(selection), let first = variants.first { selection = first }
        }
    }

    private func stat(_ title: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title.uppercased()).font(.caption2.weight(.bold)).foregroundStyle(Color.white.opacity(0.5))
            Text(value).font(.subheadline.monospacedDigit().weight(.semibold)).foregroundStyle(DF.cyan).lineLimit(1).minimumScaleFactor(0.7)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(10)
        .background(Color.black.opacity(0.18), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }

    private func runBenchmark() {
        let variant = selection
        guard let url = Bundle.main.url(forResource: "RapidNet_\(variant)", withExtension: "mlmodelc") else { return }
        isRunning = true
        errorText = nil
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let r = try Self.benchmark(variant: variant, modelURL: url)
                DispatchQueue.main.async { results[variant] = r; isRunning = false }
            } catch {
                DispatchQueue.main.async { errorText = "Benchmark failed: \(error.localizedDescription)"; isRunning = false }
            }
        }
    }

    nonisolated private static func benchmark(variant: String, modelURL: URL) throws -> BenchmarkResult {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine
        let process = ProcessInfo.processInfo
        let thermalStart = thermalName(process.thermalState)
        let rounds = 50
        let predictionsPerRound = 50
        let trim = 10
        var retained: [Double] = []

        for _ in 0..<rounds {
            try autoreleasepool {
                let model = try MLModel(contentsOf: modelURL, configuration: configuration)
                let input = try MLMultiArray(shape: [1, 3, 224, 224], dataType: .float32)
                for index in 0..<input.count { input[index] = 0 }
                let features = try MLDictionaryFeatureProvider(dictionary: ["inputs": input])

                // The first prediction in each round warms the model and is excluded.
                _ = try model.prediction(from: features)
                var timings: [Double] = []
                for _ in 1..<predictionsPerRound {
                    let start = DispatchTime.now().uptimeNanoseconds
                    _ = try model.prediction(from: features)
                    let elapsed = DispatchTime.now().uptimeNanoseconds - start
                    timings.append(Double(elapsed) / 1_000_000)
                }
                retained.append(contentsOf: timings.sorted().dropFirst(trim).dropLast(trim))
            }
        }

        let mean = retained.reduce(0, +) / Double(retained.count)
        return BenchmarkResult(variant: variant, meanMs: mean, retained: retained.count,
                               thermalStart: thermalStart, thermalEnd: thermalName(process.thermalState))
    }

    nonisolated private static func thermalName(_ state: ProcessInfo.ThermalState) -> String {
        switch state {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }
}

private struct ProbabilityBarLatency: View {
    let label: String
    let ms: Double
    let fraction: Double

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(label).font(.subheadline).foregroundStyle(Color.white)
                Spacer()
                Text(String(format: "%.3f ms", ms)).font(.caption.monospacedDigit().weight(.semibold)).foregroundStyle(Color.white.opacity(0.8))
            }
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Capsule().fill(Color.white.opacity(0.08))
                    Capsule().fill(DF.accent).frame(width: max(4, geo.size.width * CGFloat(fraction)))
                }
            }
            .frame(height: 6)
        }
    }
}
