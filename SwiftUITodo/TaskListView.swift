import CoreML
import Foundation
import SwiftUI

struct TaskListView: View {
    private let variants = ["Ti", "S", "M", "B"].filter {
        Bundle.main.url(forResource: "RapidNet_\($0)", withExtension: "mlmodelc") != nil
    }
    @State private var selection = "M"
    @State private var resultText = "50 rounds · 50 predictions per round · trim 10 at each end"
    @State private var isRunning = false
    @State private var showCamera = false
    @State private var detectsObjects = false

    var body: some View {
        VStack(spacing: 20) {
            Text("RapidNet benchmark").font(.title2.bold())
            Picker("Model", selection: $selection) {
                ForEach(variants, id: \.self) { variant in
                    Text("RapidNet-\(variant)").tag(variant)
                }
            }
            .pickerStyle(.segmented)
            .disabled(isRunning)

            Text(resultText)
                .monospacedDigit()
                .multilineTextAlignment(.center)
                .textSelection(.enabled)

            Button("Run Benchmark", action: runBenchmark)
                .buttonStyle(.borderedProminent)
                .disabled(isRunning || variants.isEmpty)

            Button("Open Segmentation Camera") {
                detectsObjects = false
                showCamera = true
            }
                .disabled(isRunning)
            Button("Open Detection Camera") {
                detectsObjects = true
                showCamera = true
            }
                .disabled(isRunning)
        }
        .padding()
        .sheet(isPresented: $showCamera) {
            NavigationStack {
                SegmentationCamera(detectsObjects: detectsObjects)
                    .navigationTitle(detectsObjects ? "Object detection" : "Semantic segmentation")
                    .navigationBarTitleDisplayMode(.inline)
                    .toolbar {
                        ToolbarItem(placement: .cancellationAction) {
                            Button("Done") { showCamera = false }
                        }
                    }
            }
        }
        .onAppear {
            if !variants.contains(selection), let first = variants.first {
                selection = first
            }
        }
    }

    private func runBenchmark() {
        let variant = selection
        guard let url = Bundle.main.url(forResource: "RapidNet_\(variant)", withExtension: "mlmodelc") else { return }
        isRunning = true
        resultText = "Running RapidNet-\(variant)…"

        DispatchQueue.global(qos: .userInitiated).async {
            let message: String
            do {
                message = try Self.benchmark(variant: variant, modelURL: url)
            } catch {
                message = "Benchmark failed: \(error.localizedDescription)"
            }
            DispatchQueue.main.async {
                resultText = message
                isRunning = false
            }
        }
    }

    nonisolated private static func benchmark(variant: String, modelURL: URL) throws -> String {
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
        let thermalEnd = thermalName(process.thermalState)
        let metrics = String(format: "%.3f ms per prediction\n%.1f predictions/s", mean, 1000 / mean)
        return "RapidNet-\(variant)\n\(metrics)\nCPU + Neural Engine · \(retained.count) retained\nThermal: \(thermalStart) → \(thermalEnd)"
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
