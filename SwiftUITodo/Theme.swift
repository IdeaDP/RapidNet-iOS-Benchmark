import SwiftUI

/// Deep Focus look: deep navy background, cyan accents (same palette as the app icon).
enum DF {
    static let navyTop = Color(red: 0.106, green: 0.165, blue: 0.420)     // #1B2A6B
    static let navy = Color(red: 0.055, green: 0.086, blue: 0.251)        // #0E1640
    static let navyBottom = Color(red: 0.027, green: 0.043, blue: 0.133)  // #070B22
    static let cyan = Color(red: 0.616, green: 0.969, blue: 1.0)          // #9DF7FF
    static let sky = Color(red: 0.263, green: 0.788, blue: 0.925)         // #43C9EC
    static let blue = Color(red: 0.176, green: 0.545, blue: 0.878)        // #2D8BE0

    static let accent = LinearGradient(colors: [sky, blue], startPoint: .topLeading, endPoint: .bottomTrailing)

    /// Colours for the 12 top-level ImageNet groups (stable order).
    static func groupColor(_ name: String) -> Color {
        switch name {
        case "Animal": return Color(red: 0.98, green: 0.62, blue: 0.27)
        case "Plant & fungus": return Color(red: 0.42, green: 0.85, blue: 0.45)
        case "Food": return Color(red: 0.98, green: 0.42, blue: 0.48)
        case "Person": return Color(red: 0.95, green: 0.80, blue: 0.40)
        case "Nature & place": return Color(red: 0.36, green: 0.80, blue: 0.70)
        case "Vehicle": return Color(red: 0.45, green: 0.62, blue: 1.0)
        case "Device & tool": return Color(red: 0.62, green: 0.55, blue: 1.0)
        case "Clothing & accessory": return Color(red: 0.93, green: 0.52, blue: 0.85)
        case "Household": return Color(red: 0.75, green: 0.70, blue: 0.55)
        case "Building & street": return Color(red: 0.60, green: 0.72, blue: 0.82)
        case "Sports & leisure": return Color(red: 0.55, green: 0.90, blue: 0.95)
        default: return Color(white: 0.7)
        }
    }

    static func groupIcon(_ name: String) -> String {
        switch name {
        case "Animal": return "pawprint.fill"
        case "Plant & fungus": return "leaf.fill"
        case "Food": return "fork.knife"
        case "Person": return "person.fill"
        case "Nature & place": return "mountain.2.fill"
        case "Vehicle": return "car.fill"
        case "Device & tool": return "wrench.and.screwdriver.fill"
        case "Clothing & accessory": return "tshirt.fill"
        case "Household": return "house.fill"
        case "Building & street": return "building.2.fill"
        case "Sports & leisure": return "sportscourt.fill"
        default: return "cube.fill"
        }
    }
}

struct DFBackground: View {
    var body: some View {
        ZStack {
            LinearGradient(colors: [DF.navyTop, DF.navy, DF.navyBottom], startPoint: .top, endPoint: .bottom)
            RadialGradient(colors: [DF.sky.opacity(0.18), .clear], center: .topTrailing, startRadius: 10, endRadius: 420)
        }
        .ignoresSafeArea()
    }
}

/// Rounded translucent card used everywhere.
struct DFCard<Content: View>: View {
    var padding: CGFloat = 16
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 12) { content }
            .padding(padding)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.white.opacity(0.07), in: RoundedRectangle(cornerRadius: 20, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 20, style: .continuous).stroke(Color.white.opacity(0.10), lineWidth: 1))
    }
}

struct DFSectionTitle: View {
    let title: String
    var icon: String? = nil

    var body: some View {
        HStack(spacing: 8) {
            if let icon { Image(systemName: icon).foregroundStyle(DF.cyan) }
            Text(title).font(.headline).foregroundStyle(Color.white)
        }
    }
}

/// Horizontal probability bar with label and percentage.
struct ProbabilityBar: View {
    let label: String
    let value: Double
    var color: Color = DF.sky
    var emphasised = false

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(label)
                    .font(emphasised ? .subheadline.weight(.semibold) : .subheadline)
                    .foregroundStyle(Color.white.opacity(emphasised ? 1 : 0.85))
                    .lineLimit(1)
                Spacer(minLength: 8)
                Text(Self.percent(value))
                    .font(.caption.monospacedDigit().weight(.semibold))
                    .foregroundStyle(Color.white.opacity(0.8))
            }
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Capsule().fill(Color.white.opacity(0.08))
                    Capsule().fill(color).frame(width: max(4, geo.size.width * CGFloat(min(max(value, 0), 1))))
                }
            }
            .frame(height: 6)
        }
    }

    static func percent(_ p: Double) -> String {
        if p >= 0.1 { return String(format: "%.1f%%", p * 100) }
        if p >= 0.001 { return String(format: "%.2f%%", p * 100) }
        return "<0.1%"
    }
}

/// Small rounded "chip" button (sizes, models).
struct DFChip: View {
    let title: String
    let selected: Bool
    var action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.subheadline.weight(.semibold).monospacedDigit())
                .padding(.horizontal, 14)
                .padding(.vertical, 8)
                .foregroundStyle(selected ? DF.navyBottom : Color.white.opacity(0.85))
                .background(selected ? AnyShapeStyle(DF.accent) : AnyShapeStyle(Color.white.opacity(0.08)), in: Capsule())
        }
        .buttonStyle(.plain)
    }
}

/// The Deep Focus logo (same motif as the app icon), drawn in SwiftUI.
struct DFLogo: View {
    var size: CGFloat = 64

    var body: some View {
        Canvas { ctx, sz in
            let c = CGPoint(x: sz.width / 2, y: sz.height / 2)
            let unit = sz.width / 1024
            ctx.fill(Path(roundedRect: CGRect(origin: .zero, size: sz), cornerRadius: sz.width * 0.225),
                     with: .linearGradient(Gradient(colors: [DF.navyTop, DF.navyBottom]),
                                           startPoint: .zero, endPoint: CGPoint(x: 0, y: sz.height)))
            let levels: [(CGFloat, CGFloat, Color)] = [(95, 30, DF.cyan), (200, 24, DF.sky), (310, 19, DF.blue)]
            for (spacing, r, col) in levels {
                for dx in -1...1 { for dy in -1...1 where !(dx == 0 && dy == 0) {
                    let p = CGPoint(x: c.x + CGFloat(dx) * spacing * unit, y: c.y + CGFloat(dy) * spacing * unit)
                    ctx.fill(Path(ellipseIn: CGRect(x: p.x - r * unit, y: p.y - r * unit, width: 2 * r * unit, height: 2 * r * unit)), with: .color(col))
                } }
            }
            ctx.fill(Path(ellipseIn: CGRect(x: c.x - 50 * unit, y: c.y - 50 * unit, width: 100 * unit, height: 100 * unit)), with: .color(.white))
        }
        .frame(width: size, height: size)
        .shadow(color: DF.sky.opacity(0.35), radius: 12)
    }
}
