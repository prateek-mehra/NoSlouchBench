import Cocoa
import Foundation

struct Args {
    var opacity: Double = 0.78
    var killSwitch: String = "Ctrl+Shift+K"
    var killMarker: String = ""
}

func parseArgs() -> Args {
    var args = Args()
    let argv = CommandLine.arguments
    var i = 1
    while i < argv.count {
        let a = argv[i]
        if a == "--opacity", i + 1 < argv.count {
            args.opacity = Double(argv[i + 1]) ?? 0.78
            i += 2
            continue
        }
        if a == "--kill-switch", i + 1 < argv.count {
            args.killSwitch = argv[i + 1]
            i += 2
            continue
        }
        if a == "--kill-marker", i + 1 < argv.count {
            args.killMarker = argv[i + 1]
            i += 2
            continue
        }
        i += 1
    }
    return args
}

final class BlockerApp: NSObject, NSApplicationDelegate {
    private let maxAlpha: CGFloat
    private let killSwitch: String
    private let killMarker: String

    private var window: NSWindow!
    private var label: NSTextField!
    private var timer: Timer?
    private var currentAlpha: CGFloat = 0.0
    private var targetAlpha: CGFloat = 0.0

    init(maxAlpha: CGFloat, killSwitch: String, killMarker: String) {
        self.maxAlpha = min(max(maxAlpha, 0.20), 0.95)
        self.killSwitch = killSwitch
        self.killMarker = killMarker
        super.init()
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        guard let screen = NSScreen.main else {
            NSApp.terminate(nil)
            return
        }

        window = NSWindow(
            contentRect: screen.frame,
            styleMask: [.borderless],
            backing: .buffered,
            defer: false
        )
        window.backgroundColor = NSColor.black
        window.isOpaque = false
        window.alphaValue = 0.0
        window.level = .screenSaver
        window.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary, .ignoresCycle]
        window.ignoresMouseEvents = false
        window.hasShadow = false

        label = NSTextField(labelWithString: "Sit straight to continue\n\nKill switch: \(killSwitch)")
        label.textColor = NSColor.white
        label.font = NSFont.boldSystemFont(ofSize: 42)
        label.alignment = .center
        label.backgroundColor = .clear
        label.isBezeled = false
        label.drawsBackground = false
        label.translatesAutoresizingMaskIntoConstraints = false

        let content = NSView(frame: screen.frame)
        content.wantsLayer = true
        content.layer?.backgroundColor = NSColor.black.cgColor
        content.addSubview(label)
        NSLayoutConstraint.activate([
            label.centerXAnchor.constraint(equalTo: content.centerXAnchor),
            label.centerYAnchor.constraint(equalTo: content.centerYAnchor)
        ])

        window.contentView = content
        window.orderOut(nil)

        NSEvent.addLocalMonitorForEvents(matching: [.keyDown]) { [weak self] event in
            guard let self = self else { return event }
            if self.isKillSwitch(event: event) {
                self.writeKillMarker()
                NSApp.terminate(nil)
                return nil
            }
            if self.currentAlpha > 0.02 {
                return nil
            }
            return event
        }

        NSEvent.addLocalMonitorForEvents(matching: [.leftMouseDown, .leftMouseUp, .rightMouseDown, .rightMouseUp, .otherMouseDown, .otherMouseUp, .scrollWheel, .mouseMoved, .leftMouseDragged, .rightMouseDragged, .otherMouseDragged]) { [weak self] event in
            guard let self = self else { return event }
            if self.currentAlpha > 0.02 {
                return nil
            }
            return event
        }

        timer = Timer.scheduledTimer(withTimeInterval: 0.025, repeats: true) { [weak self] _ in
            self?.tick()
        }

        startInputReader()
    }

    private func startInputReader() {
        DispatchQueue.global(qos: .userInteractive).async {
            while let line = readLine() {
                let cmd = line.trimmingCharacters(in: .whitespacesAndNewlines).uppercased()
                DispatchQueue.main.async {
                    self.handle(cmd: cmd)
                }
            }
            DispatchQueue.main.async {
                NSApp.terminate(nil)
            }
        }
    }

    private func handle(cmd: String) {
        switch cmd {
        case "SHOW":
            targetAlpha = maxAlpha
            window.makeKeyAndOrderFront(nil)
            NSApp.activate(ignoringOtherApps: true)
        case "HIDE":
            targetAlpha = 0.0
        case "EXIT":
            NSApp.terminate(nil)
        default:
            break
        }
    }

    private func tick() {
        if currentAlpha < targetAlpha {
            currentAlpha = min(currentAlpha + 0.05, targetAlpha)
            window.alphaValue = currentAlpha
            if currentAlpha > 0.01 {
                window.makeKeyAndOrderFront(nil)
                NSApp.activate(ignoringOtherApps: true)
            }
        } else if currentAlpha > targetAlpha {
            currentAlpha = max(currentAlpha - 0.05, targetAlpha)
            window.alphaValue = currentAlpha
            if currentAlpha <= 0.001 {
                window.orderOut(nil)
            }
        }
    }

    private func isKillSwitch(event: NSEvent) -> Bool {
        let chars = event.charactersIgnoringModifiers?.lowercased() ?? ""
        let mods = event.modifierFlags.intersection([.control, .shift, .command, .option])
        let parts = killSwitch.lowercased().split(separator: "+").map { String($0) }

        let needsCtrl = parts.contains("ctrl") || parts.contains("control")
        let needsShift = parts.contains("shift")
        let needsCmd = parts.contains("cmd") || parts.contains("command")
        let needsAlt = parts.contains("alt") || parts.contains("option")
        let key = parts.last ?? "k"

        if needsCtrl && !mods.contains(.control) { return false }
        if needsShift && !mods.contains(.shift) { return false }
        if needsCmd && !mods.contains(.command) { return false }
        if needsAlt && !mods.contains(.option) { return false }
        return chars == key
    }

    private func writeKillMarker() {
        guard !killMarker.isEmpty else { return }
        do {
            try "killed\n".write(toFile: killMarker, atomically: true, encoding: .utf8)
        } catch {
            // no-op
        }
    }
}

let args = parseArgs()
let app = NSApplication.shared
app.setActivationPolicy(.accessory)
let delegate = BlockerApp(maxAlpha: CGFloat(args.opacity), killSwitch: args.killSwitch, killMarker: args.killMarker)
app.delegate = delegate
app.run()
