import AppKit

/// Custom separator view with controllable width and color.
final class SeparatorView: NSView {
  private let line = NSView()

  /// `height` is the whole item, the hairline centered in it. 12 matches the
  /// list's density; a model page passes `pageHeight`, since its few rows sit
  /// in roomier sections.
  init(height: CGFloat = 12) {
    super.init(frame: .zero)
    translatesAutoresizingMaskIntoConstraints = false

    heightAnchor.constraint(equalToConstant: height).isActive = true
    widthAnchor.constraint(equalToConstant: Layout.menuWidth).isActive = true

    line.translatesAutoresizingMaskIntoConstraints = false
    line.wantsLayer = true

    addSubview(line)

    // Align with content (outer + inner padding)
    // This makes the separator start where the text starts, which is "wider" than standard system separator
    // and "closer to the edges" (13px vs ~16px).
    let padding = Layout.outerHorizontalPadding + Layout.innerHorizontalPadding

    NSLayoutConstraint.activate([
      line.heightAnchor.constraint(equalToConstant: 1),
      line.centerYAnchor.constraint(equalTo: centerYAnchor),
      line.leadingAnchor.constraint(equalTo: leadingAnchor, constant: padding),
      line.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -padding),
    ])
  }

  /// Separator height on a model page: 4pt more breathing room around each
  /// hairline than the list's, so the page's sections (header, actions,
  /// footer) read as separate blocks rather than one run of rows.
  static let pageHeight: CGFloat = 16

  required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }

  override func updateLayer() {
    line.layer?.backgroundColor = Theme.Colors.separator.cgColor
  }

  override func viewDidChangeEffectiveAppearance() {
    super.viewDidChangeEffectiveAppearance()
    updateLayer()
  }
}
