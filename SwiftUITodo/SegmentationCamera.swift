import AVFoundation
import CoreImage
import CoreML
import Foundation
import SwiftUI
import UIKit

struct SegmentationCamera: UIViewControllerRepresentable {
    func makeUIViewController(context: Context) -> SegmentationCameraController {
        SegmentationCameraController()
    }

    func updateUIViewController(_ controller: SegmentationCameraController, context: Context) {}
}

final class SegmentationCameraController: UIViewController {
    private let session = AVCaptureSession()
    private let captureQueue = DispatchQueue(label: "rapidnet.camera.capture", qos: .userInitiated)
    private let frames = SegmentationFrames()
    private let imageView = UIImageView()
    private let status = UILabel()

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
    private let maskSize = 128
    private let opacity: UInt8 = 110
    private var model: MLModel?
    private var input: MLMultiArray?
    private var features: MLDictionaryFeatureProvider?
    private var palette: [[UInt8]] = []
    private var failureReported = false
    private var start = ProcessInfo.processInfo.systemUptime
    private var frameCount = 0

    private struct Labels: Decodable {
        let palette: [[UInt8]]
    }

    func loadModel() throws {
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
        let tensor = try MLMultiArray(shape: [1, 3, 512, 512], dataType: .float32)
        features = try MLDictionaryFeatureProvider(dictionary: ["inputs": tensor])
        input = tensor
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard !failureReported, let buffer = CMSampleBufferGetImageBuffer(sampleBuffer),
              let model, let input, let features else { return }
        do {
            let square = prepare(buffer, into: input)
            let clock = ProcessInfo.processInfo
            let predictionStart = clock.systemUptime
            let result = try model.prediction(from: features)
            let milliseconds = (clock.systemUptime - predictionStart) * 1_000
            guard let classes = result.featureValue(for: "class_map")?.multiArrayValue,
                  let mask = makeMask(classes),
                  let image = context.createCGImage(mask.composited(over: square), from: square.extent) else {
                throw SegmentationError.message("The model returned an unexpected mask.")
            }
            let now = clock.systemUptime
            if frameCount == 0 { start = now }
            frameCount += 1
            let fps = Double(frameCount - 1) / max(now - start, 0.001)
            let metrics = String(format: "RapidNet-M · 512×512\n%.1f ms model time · %.1f processed FPS", milliseconds, fps)
            DispatchQueue.main.async { [weak self] in self?.onFrame?(image, metrics) }
        } catch {
            failureReported = true
            let message = error.localizedDescription
            DispatchQueue.main.async { [weak self] in self?.onError?(message) }
        }
    }

    private func prepare(_ buffer: CVPixelBuffer, into input: MLMultiArray) -> CIImage {
        let frame = CIImage(cvPixelBuffer: buffer)
        let extent = frame.extent
        let side = min(extent.width, extent.height)
        let crop = CGRect(x: extent.midX - side / 2, y: extent.midY - side / 2, width: side, height: side).integral
        let scale = CGFloat(inputSize) / crop.width
        let square = frame.cropped(to: crop)
            .transformed(by: CGAffineTransform(translationX: -crop.minX, y: -crop.minY))
            .transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        var pixels = [UInt8](repeating: 0, count: inputSize * inputSize * 4)
        pixels.withUnsafeMutableBytes { bytes in
            guard let address = bytes.baseAddress else { return }
            context.render(square, toBitmap: address, rowBytes: inputSize * 4,
                           bounds: square.extent, format: .RGBA8, colorSpace: colorSpace)
        }
        let values = input.dataPointer.bindMemory(to: Float.self, capacity: input.count)
        let mean: [Float] = [123.675, 116.28, 103.53]
        let deviation: [Float] = [58.395, 57.12, 57.375]
        let plane = inputSize * inputSize
        for row in 0..<inputSize {
            for col in 0..<inputSize {
                let offset = row * inputSize + col
                let rgba = ((inputSize - 1 - row) * inputSize + col) * 4
                for channel in 0..<3 {
                    values[channel * plane + offset] = (Float(pixels[rgba + channel]) - mean[channel]) / deviation[channel]
                }
            }
        }
        return square
    }

    private func makeMask(_ classes: MLMultiArray) -> CIImage? {
        guard classes.count == maskSize * maskSize,
              Array(classes.shape.map(\.intValue).suffix(2)) == [maskSize, maskSize] else { return nil }
        var rgba = Data(count: maskSize * maskSize * 4)
        rgba.withUnsafeMutableBytes { (bytes: UnsafeMutableRawBufferPointer) in
            for row in 0..<maskSize {
                for col in 0..<maskSize {
                    let id = classes[row * maskSize + col].intValue
                    guard palette.indices.contains(id) else { continue }
                    let offset = (row * maskSize + col) * 4
                    for channel in 0..<3 {
                        bytes[offset + channel] = UInt8(Int(palette[id][channel]) * Int(opacity) / 255)
                    }
                    bytes[offset + 3] = opacity
                }
            }
        }
        guard let provider = CGDataProvider(data: rgba as CFData),
              let image = CGImage(width: maskSize, height: maskSize, bitsPerComponent: 8,
                                  bitsPerPixel: 32, bytesPerRow: maskSize * 4, space: colorSpace,
                                  bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                                  provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent) else { return nil }
        let scale = CGFloat(inputSize) / CGFloat(maskSize)
        return CIImage(cgImage: image).samplingNearest().transformed(by: CGAffineTransform(scaleX: scale, y: scale))
    }
}
