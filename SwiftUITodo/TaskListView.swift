import CoreML
import Foundation
import SwiftUI

struct TaskListView: View {
    @State private var resultText = "Ready to benchmark RapidNet-M"
    @State private var isRunning = false

    var body: some View {
        VStack(spacing: 20) {
            Text(resultText)
                .font(.headline)
                .multilineTextAlignment(.center)

            Button("Run Benchmark", action: runBenchmark)
                .buttonStyle(.borderedProminent)
                .disabled(isRunning)
        }
        .padding()
    }

    private func runBenchmark() {
        isRunning = true
        resultText = "Running RapidNet-M…"

        DispatchQueue.global(qos: .userInitiated).async {
            let message: String
            do {
                let configuration = MLModelConfiguration()
                configuration.computeUnits = .all
                let model = try RapidNet_M(configuration: configuration).model

                let input = try MLMultiArray(shape: [1, 3, 224, 224], dataType: .float32)
                for index in 0..<input.count {
                    input[index] = 0
                }
                let features = try MLDictionaryFeatureProvider(dictionary: ["inputs": input])

                for _ in 0..<50 {
                    _ = try model.prediction(from: features)
                }

                let sampleCount = 100
                let start = ProcessInfo.processInfo.systemUptime
                for _ in 0..<sampleCount {
                    _ = try model.prediction(from: features)
                }
                let elapsed = ProcessInfo.processInfo.systemUptime - start
                let averageMs = elapsed * 1000 / Double(sampleCount)
                message = String(format: "Average: %.2f ms per prediction\nThroughput: %.1f predictions/s", averageMs, 1000 / averageMs)
            } catch {
                message = "Benchmark failed: \(error.localizedDescription)"
            }

            DispatchQueue.main.async {
                resultText = message
                isRunning = false
            }
        }
    }
}
