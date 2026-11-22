import PhotosUI
import SwiftUI
import UIKit

/// Image classification with RapidNet: pick/take a photo, choose model + input size,
/// see the answer as a hierarchy (Group → Subgroup → Class) and compare input sizes.
struct ClassificationView: View {
    private struct Sample: Identifiable { let id: String; let title: String }
    private let samples = [Sample(id: "Sample_Cat", title: "Cat"),
                           Sample(id: "Sample_Coffee", title: "Espresso"),
                           Sample(id: "Sample_Rocket", title: "Rocket")]

    @State private var image: UIImage?
    @State private var pickerItem: PhotosPickerItem?
    @State private var showCamera = false
    @State private var variant = RapidNetClassifier.variants.contains("M") ? "M" : (RapidNetClassifier.variants.first ?? "Ti")
    @State private var size = 224
    @State private var compareSizes = true

    @State private var result: ClassificationResult?
    @State private var sizeResults: [ClassificationResult] = []
    @State private var isRunning = false
    @State private var progress = ""
    @State private var errorText: String?
    @State private var expandedGroups: Set<String> = []
    @State private var expandedSubgroups: Set<String> = []

    var body: some View {
        ZStack {
            DFBackground()
            ScrollView {
                VStack(spacing: 16) {
                    imageCard
                    controlsCard
                    if let errorText {
                        DFCard { Label(errorText, systemImage: "exclamationmark.triangle.fill").foregroundStyle(Color.orange) }
                    }
                    if let result {
                        answerCard(result)
                        groupsCard(result)
                        treeCard(result)
                        if sizeResults.count > 1 { sizesCard }
                        allClassesLink(result)
                    }
                }
                .padding()
            }
        }
        .navigationTitle("Classify")
        .navigationBarTitleDisplayMode(.inline)
        .toolbarColorScheme(.dark, for: .navigationBar)
        .sheet(isPresented: $showCamera) {
            CameraCapture { picked in
                if let picked { setImage(picked) }
                showCamera = false
            }
            .ignoresSafeArea()
        }
        .onChange(of: pickerItem) { item in
            guard let item else { return }
            Task {
                if let data = try? await item.loadTransferable(type: Data.self), let picked = UIImage(data: data) {
                    await MainActor.run { setImage(picked) }
                }
            }
        }
    }

    // MARK: Image

    private var imageCard: some View {
        DFCard(padding: 12) {
            ZStack {
                if let image {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFill()
                        .frame(maxWidth: .infinity)
                        .frame(height: 260)
                        .clipped()
                        .overlay(alignment: .center) { cropGuide }
                } else {
                    VStack(spacing: 10) {
                        Image(systemName: "photo.on.rectangle.angled").font(.system(size: 44)).foregroundStyle(DF.cyan)
                        Text("Choose a photo to classify").font(.headline).foregroundStyle(Color.white)
                        Text("1000 ImageNet classes · grouped into 12 groups and 47 subgroups")
                            .font(.caption).foregroundStyle(Color.white.opacity(0.6)).multilineTextAlignment(.center)
                    }
                    .frame(maxWidth: .infinity)
                    .frame(height: 220)
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))

            HStack(spacing: 10) {
                PhotosPicker(selection: $pickerItem, matching: .images) {
                    sourceButtonLabel("Photos", "photo.fill")
                }
                Button { showCamera = true } label: { sourceButtonLabel("Camera", "camera.fill") }
                    .disabled(!UIImagePickerController.isSourceTypeAvailable(.camera))
                Menu {
                    ForEach(samples) { s in
                        Button(s.title) {
                            if let url = Bundle.main.url(forResource: s.id, withExtension: "jpg"),
                               let img = UIImage(contentsOfFile: url.path) { setImage(img) }
                        }
                    }
                } label: { sourceButtonLabel("Samples", "sparkles") }
            }
            .buttonStyle(.plain)
        }
    }

    /// Square outline = the centre crop the model actually sees.
    private var cropGuide: some View {
        GeometryReader { geo in
            let side = min(geo.size.width, geo.size.height) - 8
            RoundedRectangle(cornerRadius: 10)
                .stroke(DF.cyan.opacity(0.8), style: StrokeStyle(lineWidth: 2, dash: [7, 5]))
                .frame(width: side, height: side)
                .position(x: geo.size.width / 2, y: geo.size.height / 2)
        }
        .allowsHitTesting(false)
    }

    private func sourceButtonLabel(_ title: String, _ icon: String) -> some View {
        Label(title, systemImage: icon)
            .font(.subheadline.weight(.semibold))
            .frame(maxWidth: .infinity)
            .padding(.vertical, 10)
            .foregroundStyle(Color.white)
            .background(Color.white.opacity(0.10), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }

    // MARK: Controls

    private var controlsCard: some View {
        DFCard {
            DFSectionTitle(title: "Model", icon: "cpu")
            HStack(spacing: 8) {
                ForEach(RapidNetClassifier.variants, id: \.self) { v in
                    DFChip(title: "RapidNet-\(v)", selected: v == variant) { variant = v }
                }
            }
            .disabled(isRunning)

            DFSectionTitle(title: "Input size", icon: "arrow.up.left.and.arrow.down.right")
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(RapidNetClassifier.sizes, id: \.self) { s in
                        DFChip(title: "\(s) px", selected: s == size) { size = s }
                    }
                }
            }
            .disabled(isRunning)

            Toggle(isOn: $compareSizes) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Compare all input sizes").foregroundStyle(Color.white)
                    Text("Runs \(RapidNetClassifier.sizes.first ?? 128)–\(RapidNetClassifier.sizes.last ?? 384) px and shows how the answer changes")
                        .font(.caption).foregroundStyle(Color.white.opacity(0.6))
                }
            }
            .tint(DF.sky)
            .disabled(isRunning)

            Button(action: run) {
                HStack {
                    if isRunning { ProgressView().tint(DF.navyBottom) }
                    Text(isRunning ? progress : "Classify").font(.headline)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 14)
                .foregroundStyle(DF.navyBottom)
                .background(DF.accent, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            }
            .buttonStyle(.plain)
            .disabled(image == nil || isRunning || RapidNetClassifier.variants.isEmpty)
            .opacity(image == nil || RapidNetClassifier.variants.isEmpty ? 0.5 : 1)

            if RapidNetClassifier.variants.isEmpty {
                Text("No classifier models in the app. Run tools/convert_classifiers_flexible.py and rebuild.")
                    .font(.caption).foregroundStyle(Color.orange)
            }
        }
    }

    // MARK: Results

    private func answerCard(_ r: ClassificationResult) -> some View {
        let path = r.path
        return DFCard {
            HStack {
                DFSectionTitle(title: "Prediction", icon: "scope")
                Spacer()
                Text("RapidNet-\(r.variant) · \(r.size) px · \(String(format: "%.1f ms", r.latencyMs))")
                    .font(.caption.monospacedDigit()).foregroundStyle(Color.white.opacity(0.6))
            }
            Text(r.top.name)
                .font(.system(size: 30, weight: .bold, design: .rounded))
                .foregroundStyle(Color.white)
            Text("\(ProbabilityBar.percent(r.top.probability)) confidence")
                .font(.subheadline.weight(.medium)).foregroundStyle(DF.cyan)

            // Level 1 → Level 2 → Level 3 breadcrumb with probabilities
            VStack(alignment: .leading, spacing: 10) {
                levelRow(level: "Group", name: path.group.name, p: path.group.probability,
                         icon: DF.groupIcon(path.group.name), color: DF.groupColor(path.group.name))
                if let sg = path.subgroup {
                    levelRow(level: "Subgroup", name: sg.name, p: sg.probability, icon: "arrow.turn.down.right",
                             color: DF.groupColor(path.group.name).opacity(0.8))
                }
                if let c = path.cls {
                    levelRow(level: "Class", name: c.name, p: c.probability, icon: "smallcircle.filled.circle",
                             color: DF.sky)
                }
            }
            .padding(12)
            .background(Color.black.opacity(0.18), in: RoundedRectangle(cornerRadius: 14, style: .continuous))

            if let c = path.cls, c.id != r.top.id {
                Text("The single most likely class is \(r.top.name), but its group has less total probability than \(path.group.name).")
                    .font(.caption).foregroundStyle(Color.white.opacity(0.6))
            }
        }
    }

    private func levelRow(level: String, name: String, p: Double, icon: String, color: Color) -> some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .font(.system(size: 14, weight: .bold))
                .frame(width: 30, height: 30)
                .foregroundStyle(DF.navyBottom)
                .background(color, in: Circle())
            VStack(alignment: .leading, spacing: 1) {
                Text(level.uppercased()).font(.caption2.weight(.bold)).foregroundStyle(Color.white.opacity(0.5))
                Text(name).font(.subheadline.weight(.semibold)).foregroundStyle(Color.white)
            }
            Spacer()
            Text(ProbabilityBar.percent(p)).font(.subheadline.monospacedDigit().weight(.bold)).foregroundStyle(color)
        }
    }

    private func groupsCard(_ r: ClassificationResult) -> some View {
        DFCard {
            DFSectionTitle(title: "Level 1 · Groups", icon: "square.grid.3x3.fill")
            ForEach(r.groups) { g in
                ProbabilityBar(label: g.name, value: g.probability, color: DF.groupColor(g.name), emphasised: g.id == r.topGroup.id)
            }
        }
    }

    /// Full tree: every group → its subgroups → their classes (all 1000 outcomes reachable).
    private func treeCard(_ r: ClassificationResult) -> some View {
        DFCard {
            DFSectionTitle(title: "All outcomes · Group → Subgroup → Class", icon: "list.bullet.indent")
            Text("Tap to expand. Probabilities add up across levels.")
                .font(.caption).foregroundStyle(Color.white.opacity(0.6))
            ForEach(r.groups) { g in
                VStack(alignment: .leading, spacing: 8) {
                    treeRow(title: g.name, p: g.probability, count: g.classCount, depth: 0,
                            expanded: expandedGroups.contains(g.id), color: DF.groupColor(g.name)) {
                        if expandedGroups.contains(g.id) { expandedGroups.remove(g.id) } else { expandedGroups.insert(g.id) }
                    }
                    if expandedGroups.contains(g.id) {
                        ForEach(r.subgroups(of: g.name)) { sg in
                            treeRow(title: sg.name, p: sg.probability, count: sg.classCount, depth: 1,
                                    expanded: expandedSubgroups.contains(sg.id), color: DF.groupColor(g.name).opacity(0.75)) {
                                if expandedSubgroups.contains(sg.id) { expandedSubgroups.remove(sg.id) } else { expandedSubgroups.insert(sg.id) }
                            }
                            if expandedSubgroups.contains(sg.id) {
                                ForEach(r.classes(in: sg.id)) { c in
                                    ProbabilityBar(label: c.name, value: c.probability, color: DF.sky, emphasised: c.id == r.top.id)
                                        .padding(.leading, 40)
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private func treeRow(title: String, p: Double, count: Int, depth: Int, expanded: Bool, color: Color,
                         action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 10) {
                Image(systemName: expanded ? "chevron.down" : "chevron.right")
                    .font(.caption.weight(.bold)).foregroundStyle(Color.white.opacity(0.6)).frame(width: 14)
                Circle().fill(color).frame(width: 10, height: 10)
                Text(title).font(depth == 0 ? .subheadline.weight(.semibold) : .subheadline).foregroundStyle(Color.white)
                Text("\(count)").font(.caption2).foregroundStyle(Color.white.opacity(0.45))
                Spacer()
                Text(ProbabilityBar.percent(p)).font(.caption.monospacedDigit().weight(.semibold)).foregroundStyle(Color.white.opacity(0.85))
            }
            .padding(.leading, CGFloat(depth) * 20)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    private var sizesCard: some View {
        DFCard {
            DFSectionTitle(title: "Input size comparison · RapidNet-\(sizeResults.first?.variant ?? variant)", icon: "aspectratio")
            ForEach(sizeResults) { r in
                HStack(spacing: 12) {
                    Text("\(r.size)")
                        .font(.subheadline.monospacedDigit().weight(.bold))
                        .frame(width: 46, height: 34)
                        .foregroundStyle(r.size == size ? DF.navyBottom : Color.white)
                        .background(r.size == size ? AnyShapeStyle(DF.accent) : AnyShapeStyle(Color.white.opacity(0.08)),
                                    in: RoundedRectangle(cornerRadius: 8))
                    VStack(alignment: .leading, spacing: 2) {
                        Text(r.top.name).font(.subheadline.weight(.semibold)).foregroundStyle(Color.white).lineLimit(1)
                        Text("\(r.topGroup.name) › \(r.path.subgroup?.name ?? "")")
                            .font(.caption).foregroundStyle(Color.white.opacity(0.55)).lineLimit(1)
                    }
                    Spacer()
                    VStack(alignment: .trailing, spacing: 2) {
                        Text(ProbabilityBar.percent(r.top.probability)).font(.subheadline.monospacedDigit().weight(.bold)).foregroundStyle(DF.cyan)
                        Text(String(format: "%.1f ms", r.latencyMs)).font(.caption.monospacedDigit()).foregroundStyle(Color.white.opacity(0.55))
                    }
                }
            }
            let names = Set(sizeResults.map { $0.top.name })
            Text(names.count == 1 ? "Same answer at every input size." :
                    "The answer changes with input size (\(names.count) different top classes).")
                .font(.caption).foregroundStyle(Color.white.opacity(0.6))
        }
    }

    private func allClassesLink(_ r: ClassificationResult) -> some View {
        NavigationLink {
            AllClassesView(result: r)
        } label: {
            HStack {
                Image(systemName: "list.number").foregroundStyle(DF.cyan)
                Text("All 1000 classes, ranked").foregroundStyle(Color.white).font(.subheadline.weight(.semibold))
                Spacer()
                Image(systemName: "chevron.right").foregroundStyle(Color.white.opacity(0.5))
            }
            .padding(16)
            .background(Color.white.opacity(0.07), in: RoundedRectangle(cornerRadius: 20, style: .continuous))
        }
    }

    // MARK: Actions

    private func setImage(_ picked: UIImage) {
        image = picked
        result = nil
        sizeResults = []
        errorText = nil
    }

    private func run() {
        guard let image, let cg = image.uprightCGImage() else { return }
        let v = variant, mainSize = size, all = compareSizes
        let sizes = all ? RapidNetClassifier.sizes : [mainSize]
        isRunning = true
        errorText = nil
        progress = "Loading RapidNet-\(v)…"
        DispatchQueue.global(qos: .userInitiated).async {
            var results: [ClassificationResult] = []
            var failure: String?
            for s in sizes {
                DispatchQueue.main.async { progress = "RapidNet-\(v) at \(s) px…" }
                do { results.append(try RapidNetClassifier.shared.classify(cg, variant: v, size: s)) }
                catch { failure = error.localizedDescription; break }
            }
            DispatchQueue.main.async {
                isRunning = false
                if let failure { errorText = failure; return }
                sizeResults = results
                result = results.first { $0.size == mainSize } ?? results.first
                if let top = result?.path {
                    expandedGroups = [top.group.id]
                    expandedSubgroups = top.subgroup.map { [$0.id] } ?? []
                }
            }
        }
    }
}

/// Ranked list of all 1000 classes with search.
struct AllClassesView: View {
    let result: ClassificationResult
    @State private var query = ""

    var body: some View {
        let rank = Dictionary(uniqueKeysWithValues: result.classes.enumerated().map { ($0.element.id, $0.offset + 1) })
        let filtered = query.isEmpty ? result.classes : result.classes.filter {
            $0.name.localizedCaseInsensitiveContains(query) || $0.level2.localizedCaseInsensitiveContains(query) ||
            $0.level1.localizedCaseInsensitiveContains(query)
        }
        List {
            ForEach(filtered) { c in
                HStack(spacing: 12) {
                    Text("\(rank[c.id] ?? 0)")
                        .font(.caption.monospacedDigit()).foregroundStyle(.secondary).frame(width: 38, alignment: .trailing)
                    Circle().fill(DF.groupColor(c.level1)).frame(width: 8, height: 8)
                    VStack(alignment: .leading, spacing: 1) {
                        Text(c.name).font(.subheadline.weight(.medium))
                        Text("\(c.level1) › \(c.level2)").font(.caption).foregroundStyle(.secondary)
                    }
                    Spacer()
                    Text(ProbabilityBar.percent(c.probability)).font(.caption.monospacedDigit())
                }
            }
        }
        .searchable(text: $query, prompt: "Search class or group")
        .navigationTitle("RapidNet-\(result.variant) · \(result.size) px")
        .navigationBarTitleDisplayMode(.inline)
    }
}

/// Take a photo with the system camera UI.
struct CameraCapture: UIViewControllerRepresentable {
    var onFinish: (UIImage?) -> Void

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.sourceType = .camera
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}
    func makeCoordinator() -> Coordinator { Coordinator(onFinish: onFinish) }

    final class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        let onFinish: (UIImage?) -> Void
        init(onFinish: @escaping (UIImage?) -> Void) { self.onFinish = onFinish }
        func imagePickerController(_ picker: UIImagePickerController,
                                   didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]) {
            onFinish(info[.originalImage] as? UIImage)
        }
        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) { onFinish(nil) }
    }
}
