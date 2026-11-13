import CoreGraphics
import CoreML
import Foundation

final class RapidNetDetector {
    struct Detection {
        let box: CGRect
        let label: String
        let score: Float
    }

    private struct Box {
        let left: Float
        let top: Float
        let right: Float
        let bottom: Float

        var width: Float { max(0, right - left) }
        var height: Float { max(0, bottom - top) }
        var rect: CGRect {
            CGRect(x: CGFloat(left), y: CGFloat(top), width: CGFloat(width), height: CGFloat(height))
        }

        func decoded(_ dx: Float, _ dy: Float, _ dw: Float, _ dh: Float) -> Box {
            let x = (left + right) * 0.5 + dx * width
            let y = (top + bottom) * 0.5 + dy * height
            let w = width * exp(max(-4.135, min(4.135, dw))) * 0.5
            let h = height * exp(max(-4.135, min(4.135, dh))) * 0.5
            return Box(left: Self.clip(x - w), top: Self.clip(y - h),
                       right: Self.clip(x + w), bottom: Self.clip(y + h))
        }

        private static func clip(_ value: Float) -> Float { max(0, min(512, value)) }

        func overlap(_ other: Box) -> Float {
            let x = max(0, min(right, other.right) - max(left, other.left))
            let y = max(0, min(bottom, other.bottom) - max(top, other.top))
            let intersection = x * y
            return intersection / max(width * height + other.width * other.height - intersection, 0.001)
        }
    }

    private struct Candidate {
        let box: Box
        let score: Float
        let label: Int
    }

    private struct Tensor {
        // Retain the owning array for the lifetime of the pointer.
        let array: MLMultiArray
        let pointer: UnsafePointer<Float>
        let shape: [Int]
        let strides: [Int]

        init(_ result: MLFeatureProvider, _ name: String, shape: [Int]) throws {
            guard let array = result.featureValue(for: name)?.multiArrayValue,
                  array.dataType == .float32, array.shape.map(\.intValue) == shape else {
                throw DetectorError.invalidOutput(name)
            }
            self.array = array
            self.shape = shape
            self.strides = array.strides.map(\.intValue)
            self.pointer = UnsafePointer(array.dataPointer.assumingMemoryBound(to: Float.self))
        }

        subscript(_ channel: Int, _ row: Int, _ col: Int) -> Float {
            pointer[channel * strides[1] + row * strides[2] + col * strides[3]]
        }

        subscript(_ row: Int, _ col: Int) -> Float {
            pointer[row * strides[0] + col * strides[1]]
        }
    }

    private enum DetectorError: LocalizedError {
        case missingAsset(String), invalidOutput(String)

        var errorDescription: String? {
            switch self {
            case .missingAsset(let name): return "Missing detector asset: \(name)."
            case .invalidOutput(let name): return "Unexpected detector output: \(name)."
            }
        }
    }

    private static let regionLimit = 64
    private let featureModel: MLModel
    private let head: MLModel
    private let labels: [String]
    private let regions: MLMultiArray
    private let headInput: MLDictionaryFeatureProvider

    init() throws {
        func url(_ name: String, _ suffix: String) throws -> URL {
            guard let asset = Bundle.main.url(forResource: name, withExtension: suffix) else {
                throw DetectorError.missingAsset(name)
            }
            return asset
        }
        let config = MLModelConfiguration()
        config.computeUnits = .all
        featureModel = try MLModel(contentsOf: url("RapidNet_M_DetectorFeatures", "mlmodelc"), configuration: config)
        head = try MLModel(contentsOf: url("RapidNet_M_BoxHead", "mlmodelc"), configuration: config)
        let labelFile = try url("RapidNet_COCO", "json")
        labels = try JSONDecoder().decode([String].self, from: Data(contentsOf: labelFile))
        guard labels.count == 80 else { throw DetectorError.invalidOutput("COCO class labels") }
        let tensor = try MLMultiArray(shape: [NSNumber(value: Self.regionLimit), 256, 7, 7], dataType: .float32)
        regions = tensor
        headInput = try MLDictionaryFeatureProvider(dictionary: ["regions": tensor])
    }

    func predict(_ input: MLFeatureProvider) throws -> [Detection] {
        let result = try featureModel.prediction(from: input)
        var proposals: [Candidate] = []
        var pyramid: [Tensor] = []
        for level in 0..<5 {
            let stride = 4 << level
            let size = 512 / stride
            let rpn = try Tensor(result, "rpn\(level)", shape: [1, 15, size, size])
            proposals += suppress(proposalsAtLevel(rpn, stride: stride), threshold: 0.7, limit: Self.regionLimit)
            if level < 4 { pyramid.append(try Tensor(result, "p\(level)", shape: [1, 256, size, size])) }
        }
        let selected = Array(proposals.sorted { $0.score > $1.score }.prefix(Self.regionLimit))
        guard !selected.isEmpty else { return [] }
        pool(selected, pyramid: pyramid)
        let prediction = try head.prediction(from: headInput)
        let scores = try Tensor(prediction, "scores", shape: [Self.regionLimit, 81])
        let deltas = try Tensor(prediction, "deltas", shape: [Self.regionLimit, 320])
        var candidates: [Candidate] = []
        for (row, proposal) in selected.enumerated() {
            for label in labels.indices where scores[row, label] >= 0.45 {
                let index = label * 4
                let box = proposal.box.decoded(deltas[row, index] * 0.1, deltas[row, index + 1] * 0.1,
                                               deltas[row, index + 2] * 0.2, deltas[row, index + 3] * 0.2)
                if box.width > 1, box.height > 1 {
                    candidates.append(Candidate(box: box, score: scores[row, label], label: label))
                }
            }
        }
        return suppress(candidates, threshold: 0.5, limit: 12).map {
            Detection(box: $0.box.rect, label: labels[$0.label], score: $0.score)
        }
    }

    private func proposalsAtLevel(_ tensor: Tensor, stride: Int) -> [Candidate] {
        let size = tensor.shape[2]
        let ratios: [Float] = [0.5, 1, 2]
        let step = Float(stride)
        let base = step * 8
        let anchorSizes = ratios.map { ratio in (base / sqrt(ratio), base * sqrt(ratio)) }
        // Sorting logits gives the same order as sorting their sigmoid scores.
        var ranked: [(Float, Int, Int, Int)] = []
        ranked.reserveCapacity(size * size * 3)
        for row in 0..<size {
            for col in 0..<size {
                for anchor in 0..<3 {
                    let score = tensor[anchor, row, col]
                    if score.isFinite { ranked.append((score, row, col, anchor)) }
                }
            }
        }
        ranked.sort { $0.0 > $1.0 }
        return ranked.prefix(256).compactMap { score, row, col, anchor in
            let (width, height) = anchorSizes[anchor]
            let x = Float(col) * step
            let y = Float(row) * step
            let baseBox = Box(left: x - width / 2, top: y - height / 2,
                              right: x + width / 2, bottom: y + height / 2)
            let index = 3 + anchor * 4
            let box = baseBox.decoded(tensor[index, row, col], tensor[index + 1, row, col],
                                      tensor[index + 2, row, col], tensor[index + 3, row, col])
            guard box.width > 1, box.height > 1 else { return nil }
            return Candidate(box: box, score: score, label: 0)
        }
    }

    private func suppress(_ candidates: [Candidate], threshold: Float, limit: Int) -> [Candidate] {
        var kept: [Candidate] = []
        for candidate in candidates.sorted(by: { $0.score > $1.score }) {
            if kept.contains(where: { $0.label == candidate.label && $0.box.overlap(candidate.box) > threshold }) {
                continue
            }
            kept.append(candidate)
            if kept.count == limit { break }
        }
        return kept
    }

    private func pool(_ proposals: [Candidate], pyramid: [Tensor]) {
        let output = regions.dataPointer.assumingMemoryBound(to: Float.self)
        // MLMultiArray is allocated as contiguous NCHW. The unused rows are cleared before inference.
        output.update(repeating: 0, count: regions.count)
        for (region, proposal) in proposals.enumerated() {
            let box = proposal.box
            let scale = sqrt(box.width * box.height)
            let level = max(0, min(3, Int(floor(log2(scale / 56 + 0.000001)))))
            let tensor = pyramid[level]
            let stride = Float(4 << level)
            let binX = box.width / stride / 7
            let binY = box.height / stride / 7
            let samplesX = max(1, Int(ceil(binX)))
            let samplesY = max(1, Int(ceil(binY)))
            let left = box.left / stride - 0.5
            let top = box.top / stride - 0.5
            let limit = Float(tensor.shape[2] - 1)
            let divisor = Float(samplesX * samplesY)

            // Aligned RoIAlign with adaptive bilinear sampling, as used by the checkpoint.
            for row in 0..<7 {
                for col in 0..<7 {
                    for sy in 0..<samplesY {
                        let y = min(limit, max(0, top + (Float(row) + (Float(sy) + 0.5) / Float(samplesY)) * binY))
                        let y0 = Int(y), y1 = min(y0 + 1, Int(limit))
                        let fy = y - Float(y0)
                        for sx in 0..<samplesX {
                            let x = min(limit, max(0, left + (Float(col) + (Float(sx) + 0.5) / Float(samplesX)) * binX))
                            let x0 = Int(x), x1 = min(x0 + 1, Int(limit))
                            let fx = x - Float(x0)
                            let offset = region * 256 * 49 + row * 7 + col
                            let a = (1 - fy) * (1 - fx) / divisor
                            let b = (1 - fy) * fx / divisor
                            let c = fy * (1 - fx) / divisor
                            let d = fy * fx / divisor
                            for channel in 0..<256 {
                                let upper = a * tensor[channel, y0, x0] + b * tensor[channel, y0, x1]
                                let lower = c * tensor[channel, y1, x0] + d * tensor[channel, y1, x1]
                                output[offset + channel * 49] += upper + lower
                            }
                        }
                    }
                }
            }
        }
    }
}
