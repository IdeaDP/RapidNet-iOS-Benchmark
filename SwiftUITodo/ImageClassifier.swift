import CoreGraphics
import CoreML
import Foundation
import UIKit
import Vision

// MARK: - ImageNet hierarchy (Group → Subgroup → Class), from RapidNet_ImageNet_Hierarchy.json

struct ImageNetClass: Decodable {
    let index: Int
    let name: String
    let level1: String
    let level2: String
}

final class ImageNetHierarchy {
    static let shared = ImageNetHierarchy()
    let classes: [ImageNetClass]

    private init() {
        struct File: Decodable { let classes: [ImageNetClass] }
        if let url = Bundle.main.url(forResource: "RapidNet_ImageNet_Hierarchy", withExtension: "json"),
           let data = try? Data(contentsOf: url),
           let file = try? JSONDecoder().decode(File.self, from: data) {
            classes = file.classes.sorted { $0.index < $1.index }
        } else {
            classes = (0..<1000).map { ImageNetClass(index: $0, name: "Class \($0)", level1: "Object", level2: "Other object") }
        }
    }
}

// MARK: - Results

struct ClassScore: Identifiable {
    let id: Int            // ImageNet index
    let name: String
    let level1: String
    let level2: String
    let probability: Double
}

struct GroupScore: Identifiable {
    let id: String         // "Animal" or "Animal/Dog"
    let name: String
    let parent: String?
    let probability: Double
    let classCount: Int
}

struct ClassificationResult: Identifiable {
    let id = UUID()
    let variant: String
    let size: Int
    let latencyMs: Double
    let classes: [ClassScore]        // all 1000, most likely first
    let groups: [GroupScore]         // level 1, most likely first
    let subgroups: [GroupScore]      // level 2, most likely first

    var top: ClassScore { classes[0] }
    var topGroup: GroupScore { groups[0] }
    func subgroups(of group: String) -> [GroupScore] { subgroups.filter { $0.parent == group } }
    func classes(in subgroupID: String) -> [ClassScore] { classes.filter { "\($0.level1)/\($0.level2)" == subgroupID } }
    /// Most likely subgroup inside the most likely group (a consistent top-down path).
    var path: (group: GroupScore, subgroup: GroupScore?, cls: ClassScore?) {
        let g = topGroup
        let sg = subgroups(of: g.name).first
        let c = sg.flatMap { classes(in: $0.id).first }
        return (g, sg, c)
    }
}

enum ClassifierError: LocalizedError {
    case modelMissing(String), badOutput
    var errorDescription: String? {
        switch self {
        case .modelMissing(let v): return "RapidNet_\(v)_Classifier.mlmodel is not in the app."
        case .badOutput: return "The model returned an unexpected output."
        }
    }
}

// MARK: - Engine

/// Runs the flexible-input RapidNet classifiers (RapidNet_{Ti,S,M,B}_Classifier.mlmodel).
/// Thread-safe: call from a background queue.
final class RapidNetClassifier {
    static let shared = RapidNetClassifier()
    static let sizes = [128, 160, 192, 224, 256, 288, 320, 384]
    static let variants: [String] = ["Ti", "S", "M", "B"].filter {
        Bundle.main.url(forResource: "RapidNet_\($0)_Classifier", withExtension: "mlmodelc") != nil
    }

    private var models: [String: MLModel] = [:]
    private let lock = NSLock()

    private func model(_ variant: String) throws -> MLModel {
        lock.lock(); defer { lock.unlock() }
        if let m = models[variant] { return m }
        guard let url = Bundle.main.url(forResource: "RapidNet_\(variant)_Classifier", withExtension: "mlmodelc") else {
            throw ClassifierError.modelMissing(variant)
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        let m = try MLModel(contentsOf: url, configuration: configuration)
        models[variant] = m
        return m
    }

    /// Classifies `image` with RapidNet-`variant` at `size`×`size` input (centre crop, like ImageNet evaluation).
    func classify(_ image: CGImage, variant: String, size: Int) throws -> ClassificationResult {
        let model = try model(variant)
        let input = try MLFeatureValue(
            cgImage: image, pixelsWide: size, pixelsHigh: size,
            pixelFormatType: kCVPixelFormatType_32BGRA,
            options: [.cropAndScale: VNImageCropAndScaleOption.centerCrop.rawValue])
        let provider = try MLDictionaryFeatureProvider(dictionary: ["image": input])

        _ = try model.prediction(from: provider)                  // warm-up (first run at a new size is slower)
        var timings: [Double] = []
        var output: MLFeatureProvider?
        for _ in 0..<3 {
            let start = DispatchTime.now().uptimeNanoseconds
            output = try model.prediction(from: provider)
            timings.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000)
        }
        guard let probs = output?.featureValue(for: "classProbs")?.dictionaryValue else { throw ClassifierError.badOutput }

        let info = ImageNetHierarchy.shared.classes
        var p = [Double](repeating: 0, count: info.count)
        for (key, value) in probs {
            if let k = key as? String, let i = Int(k), i < p.count { p[i] = value.doubleValue }
            else if let i = key as? Int, i < p.count { p[i] = value.doubleValue }
        }
        return Self.makeResult(variant: variant, size: size, latency: timings.sorted()[1], probabilities: p, info: info)
    }

    static func makeResult(variant: String, size: Int, latency: Double, probabilities p: [Double], info: [ImageNetClass]) -> ClassificationResult {
        let classes = info.map { ClassScore(id: $0.index, name: $0.name, level1: $0.level1, level2: $0.level2, probability: p[$0.index]) }
            .sorted { $0.probability > $1.probability }
        var g: [String: (Double, Int)] = [:]
        var sg: [String: (String, String, Double, Int)] = [:]
        for c in info {
            let prob = p[c.index]
            g[c.level1, default: (0, 0)].0 += prob
            g[c.level1, default: (0, 0)].1 += 1
            let key = "\(c.level1)/\(c.level2)"
            let old = sg[key] ?? (c.level1, c.level2, 0, 0)
            sg[key] = (old.0, old.1, old.2 + prob, old.3 + 1)
        }
        let groups = g.map { GroupScore(id: $0.key, name: $0.key, parent: nil, probability: $0.value.0, classCount: $0.value.1) }
            .sorted { $0.probability > $1.probability }
        let subgroups = sg.map { GroupScore(id: $0.key, name: $0.value.1, parent: $0.value.0, probability: $0.value.2, classCount: $0.value.3) }
            .sorted { $0.probability > $1.probability }
        return ClassificationResult(variant: variant, size: size, latencyMs: latency,
                                    classes: classes, groups: groups, subgroups: subgroups)
    }
}

extension UIImage {
    /// Upright CGImage (applies the EXIF orientation of photos), capped at `maxSide` pixels.
    func uprightCGImage(maxSide: CGFloat = 1024) -> CGImage? {
        let scale = min(1, maxSide / max(size.width, size.height))
        let target = CGSize(width: max(1, size.width * scale), height: max(1, size.height * scale))
        let format = UIGraphicsImageRendererFormat.default()
        format.scale = 1
        return UIGraphicsImageRenderer(size: target, format: format).image { _ in
            draw(in: CGRect(origin: .zero, size: target))
        }.cgImage
    }
}
