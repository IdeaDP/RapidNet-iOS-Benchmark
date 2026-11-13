import AVFoundation
import CoreImage
import CoreML
import Foundation
import SwiftUI
import UIKit

struct SegmentationCamera: UIViewControllerRepresentable {
    var detectsObjects = false

    func makeUIViewController(context: Context) -> SegmentationCameraController {
        SegmentationCameraController(detectsObjects: detectsObjects)
    }

    func updateUIViewController(_ controller: SegmentationCameraController, context: Context) {}
}

final class SegmentationCameraController: UIViewController {
    private let session = AVCaptureSession()
    private let captureQueue = DispatchQueue(label: "rapidnet.camera.capture", qos: .userInitiated)
    private let frames: SegmentationFrames
    private let imageView = UIImageView()
    private let status = UILabel()

    init(detectsObjects: Bool) {
        frames = SegmentationFrames(detectsObjects: detectsObjects)
        super.init(nibName: nil, bundle: nil)
    }

    required init?(coder: NSCoder) { nil }

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .systemBackground
        imageView.contentMode = .scaleAspectFit
        status.textAlignment = .center
        status.numberOfLines = 0
        status.font = .monospacedDigitSystemFont(ofSize: 15, weight: .regular)
        status.text = "Opening rear camera…"

        let stack = UIStackView(arrangedSubviews: [imageView, status])
        stack.axis = .vertical
        stack.spacing = 12
        stack.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(stack)
        let safe = view.safeAreaLayoutGuide
        NSLayoutConstraint.activate([
            stack.leadingAnchor.constraint(equalTo: safe.leadingAnchor, constant: 12),
            stack.trailingAnchor.constraint(equalTo: safe.trailingAnchor, constant: -12),
            stack.topAnchor.constraint(equalTo: safe.topAnchor, constant: 12),
            stack.bottomAnchor.constraint(equalTo: safe.bottomAnchor, constant: -12),
        ])
        status.setContentCompressionResistancePriority(.required, for: .vertical)

        frames.onFrame = { [weak self] image, metrics in
            guard let self, self.viewIfLoaded?.window != nil else { return }
            self.imageView.image = UIImage(cgImage: image)
            self.status.text = metrics
        }
        frames.onError = { [weak self] message in self?.status.text = message }

        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized: startCamera()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                DispatchQueue.main.async {
                    if granted { self?.startCamera() }
                    else { self?.showCameraDenied() }
                }
            }
        default: showCameraDenied()
        }
    }

    override func viewDidDisappear(_ animated: Bool) {
        super.viewDidDisappear(animated)
        let capture = session
        captureQueue.async { capture.stopRunning() }
    }

    private func showCameraDenied() {
        status.text = "Allow camera access in Settings → Apps → SwiftUITodo → Camera, then reopen this screen."
    }

    private func startCamera() {
        let capture = session
        let queue = captureQueue
        let frameProcessor = frames
        captureQueue.async { [weak self] in
            guard self != nil else { return }
            do {
                try frameProcessor.loadModel()
                guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
                    throw SegmentationError.message("The rear camera is unavailable.")
                }
                let input = try AVCaptureDeviceInput(device: camera)
                let output = AVCaptureVideoDataOutput()
                output.alwaysDiscardsLateVideoFrames = true
                output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
                output.setSampleBufferDelegate(frameProcessor, queue: queue)

                capture.beginConfiguration()
                if capture.canSetSessionPreset(.vga640x480) { capture.sessionPreset = .vga640x480 }
                guard capture.canAddInput(input), capture.canAddOutput(output) else {
                    capture.commitConfiguration()
                    throw SegmentationError.message("Could not configure the camera stream.")
                }
                capture.addInput(input)
                capture.addOutput(output)
                if let connection = output.connection(with: .video), connection.isVideoOrientationSupported {
                    connection.videoOrientation = .portrait
                }
                capture.commitConfiguration()
                capture.startRunning()
            } catch {
                let message = error.localizedDescription
                DispatchQueue.main.async { [weak self] in self?.status.text = message }
            }
        }
    }
}

private enum SegmentationError: LocalizedError {
    case message(String)

    var errorDescription: String? {
        switch self { case .message(let text): return text }
    }
}

private final class SegmentationFrames: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    var onFrame: ((CGImage, String) -> Void)?
    var onError: ((String) -> Void)?
    private let context = CIContext()
    private let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
    private let inputSize = 512
    private let opacity: UInt8 = 110
    private var model: MLModel?
    private var detector: RapidNetDetector?
    private let detectsObjects: Bool
    private var input: MLMultiArray?
    private var features: MLDictionaryFeatureProvider?
    private var palette: [[UInt8]] = []
    private var failureReported = false
    private var start = ProcessInfo.processInfo.systemUptime
    private var frameCount = 0

    private struct Labels: Decodable {
        let palette: [[UInt8]]
    }

    init(detectsObjects: Bool) {
        self.detectsObjects = detectsObjects
    }

    func loadModel() throws {
        let tensor = try MLMultiArray(shape: [1, 3, 512, 512], dataType: .float32)
        features = try MLDictionaryFeatureProvider(dictionary: ["inputs": tensor])
        input = tensor
        if detectsObjects {
            detector = try RapidNetDetector()
            return
        }
        guard let url = Bundle.main.url(forResource: "RapidNet_M_Semantic", withExtension: "mlmodelc"),
              let colors = Bundle.main.url(forResource: "RapidNet_ADE20K", withExtension: "json") else {
            throw SegmentationError.message("The segmentation model or its color palette is missing.")
        }
        let config = MLModelConfiguration()
        config.computeUnits = .all
        model = try MLModel(contentsOf: url, configuration: config)
        palette = try JSONDecoder().decode(Labels.self, from: Data(contentsOf: colors)).palette
        guard palette.count == 150, palette.allSatisfy({ $0.count == 3 }) else {
            throw SegmentationError.message("The segmentation color palette is invalid.")
        }
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard !failureReported, let buffer = CMSampleBufferGetImageBuffer(sampleBuffer),
              let input, let features else { return }
        do {
            guard let square = prepare(buffer, into: input) else {
                throw SegmentationError.message("Could not read the camera frame.")
            }
            let clock = ProcessInfo.processInfo
            let predictionStart = clock.systemUptime
            let image: CGImage
            let milliseconds: Double
            if let detector {
                let detections = try detector.predict(features)
                milliseconds = (clock.systemUptime - predictionStart) * 1_000
                guard let rendered = drawBoxes(detections, over: square) else {
                    throw SegmentationError.message("Could not draw the detection preview.")
                }
                image = rendered
            } else {
                guard let model else { return }
                let result = try model.prediction(from: features)
                milliseconds = (clock.systemUptime - predictionStart) * 1_000
                guard let classes = result.featureValue(for: "class_map")?.multiArrayValue,
                      let mask = makeMask(classes),
                      let rendered = context.createCGImage(mask.composited(over: square), from: square.extent) else {
                    throw SegmentationError.message("The model returned an unexpected mask.")
                }
                image = rendered
            }
            let now = clock.systemUptime
            if frameCount == 0 { start = now }
            frameCount += 1
            let fps = Double(frameCount - 1) / max(now - start, 0.001)
            let timing = detectsObjects ? "detector time" : "model time"
            let metrics = String(format: "RapidNet-M · 512×512\n%.1f ms \(timing) · %.1f processed FPS", milliseconds, fps)
            DispatchQueue.main.async { [weak self] in self?.onFrame?(image, metrics) }
        } catch {
            failureReported = true
            let message = error.localizedDescription
            DispatchQueue.main.async { [weak self] in self?.onError?(message) }
        }
    }

    private func drawBoxes(_ detections: [RapidNetDetector.Detection], over frame: CIImage) -> CGImage? {
        guard let image = context.createCGImage(frame, from: frame.extent) else { return nil }
        let bounds = CGRect(x: 0, y: 0, width: image.width, height: image.height)
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = true
        return UIGraphicsImageRenderer(bounds: bounds, format: format).image { renderer in
            UIImage(cgImage: image).draw(in: bounds)
            let font = UIFont.boldSystemFont(ofSize: 15)
            let color = UIColor.systemYellow
            for detection in detections {
                renderer.cgContext.setStrokeColor(color.cgColor)
                renderer.cgContext.setLineWidth(3)
                renderer.cgContext.stroke(detection.box.insetBy(dx: 1.5, dy: 1.5))
                let label = "\(detection.label) \(Int((detection.score * 100).rounded()))%" as NSString
                let attributes: [NSAttributedString.Key: Any] = [.font: font, .foregroundColor: UIColor.black]
                let size = label.size(withAttributes: attributes)
                let x = min(detection.box.minX, bounds.width - size.width - 8)
                let y = max(0, detection.box.minY - size.height - 5)
                color.setFill()
                renderer.fill(CGRect(x: x, y: y, width: size.width + 8, height: size.height + 5))
                label.draw(at: CGPoint(x: x + 4, y: y + 2), withAttributes: attributes)
            }
        }.cgImage
    }

    private func prepare(_ buffer: CVPixelBuffer, into input: MLMultiArray) -> CIImage? {
        let frame = CIImage(cvPixelBuffer: buffer)
        let extent = frame.extent
        let side = min(extent.width, extent.height)
        let crop = CGRect(x: extent.midX - side / 2, y: extent.midY - side / 2, width: side, height: side).integral
        let scale = CGFloat(inputSize) / crop.width
        let square = frame.cropped(to: crop)
            .transformed(by: CGAffineTransform(translationX: -crop.minX, y: -crop.minY))
            .transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        guard let image = context.createCGImage(square, from: square.extent) else { return nil }
        var pixels = [UInt8](repeating: 0, count: inputSize * inputSize * 4)
        let didDraw = pixels.withUnsafeMutableBytes { bytes -> Bool in
            let bitmapInfo = CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue
            guard let bitmap = CGContext(data: bytes.baseAddress, width: inputSize, height: inputSize,
                                         bitsPerComponent: 8, bytesPerRow: inputSize * 4,
                                         space: colorSpace, bitmapInfo: bitmapInfo) else { return false }
            bitmap.draw(image, in: CGRect(x: 0, y: 0, width: CGFloat(inputSize), height: CGFloat(inputSize)))
            return true
        }
        guard didDraw else { return nil }
        let values = input.dataPointer.bindMemory(to: Float.self, capacity: input.count)
        let mean: [Float] = [123.675, 116.28, 103.53]
        let deviation: [Float] = [58.395, 57.12, 57.375]
        let plane = inputSize * inputSize
        // Use the same upright image for the model and the preview.
        for channel in 0..<3 {
            for offset in 0..<plane {
                values[channel * plane + offset] = (Float(pixels[offset * 4 + channel]) - mean[channel]) / deviation[channel]
            }
        }
        return CIImage(cgImage: image)
    }

    private func makeMask(_ classes: MLMultiArray) -> CIImage? {
        let maskSize = inputSize
        guard classes.dataType == .float32, classes.count == maskSize * maskSize,
              Array(classes.shape.map(\.intValue).suffix(2)) == [maskSize, maskSize] else { return nil }
        let strides = Array(classes.strides.map(\.intValue).suffix(2))
        guard strides.count == 2, strides.allSatisfy({ $0 > 0 }) else { return nil }
        let capacity = (maskSize - 1) * (strides[0] + strides[1]) + 1
        let values = classes.dataPointer.bindMemory(to: Float.self, capacity: capacity)
        let colors = palette.map { color in color.map { UInt8(Int($0) * Int(opacity) / 255) } }
        var rgba = Data(count: maskSize * maskSize * 4)
        rgba.withUnsafeMutableBytes { (bytes: UnsafeMutableRawBufferPointer) in
            for row in 0..<maskSize {
                for col in 0..<maskSize {
                    let value = values[row * strides[0] + col * strides[1]]
                    guard value.isFinite, value >= 0, value < Float(colors.count) else { continue }
                    let color = colors[Int(value)]
                    let offset = (row * maskSize + col) * 4
                    bytes[offset] = color[0]
                    bytes[offset + 1] = color[1]
                    bytes[offset + 2] = color[2]
                    bytes[offset + 3] = opacity
                }
            }
        }
        guard let provider = CGDataProvider(data: rgba as CFData),
              let image = CGImage(width: maskSize, height: maskSize, bitsPerComponent: 8,
                                  bitsPerPixel: 32, bytesPerRow: maskSize * 4, space: colorSpace,
                                  bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                                  provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent) else { return nil }
        return CIImage(cgImage: image)
    }
}
