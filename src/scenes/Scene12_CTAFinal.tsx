import React from "react";
import {
  AbsoluteFill,
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { C, FONT, gridStyle, glowBlueCTA } from "../styles";

// Scene 12 — CTA Final (duration 254 frames)
// Phase 1 — Impact recap: local 0–100
// Phase 2 — CTA button: local 100–200
// Phase 3 — Outro: local 200–254

export const Scene12_CTAFinal: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // ========== PHASE 1: Impact Recap (0–120) ==========
  const phase1Out = interpolate(frame, [100, 118], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Intro line — local 5
  const introOp = interpolate(frame, [5, 18], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Line 1 — local 15
  const l1Scale = spring({
    fps,
    frame: Math.max(0, frame - 15),
    config: { stiffness: 160, damping: 16 },
  });
  const l1Op = interpolate(frame, [15, 28], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Line 2 — local 45
  const l2Scale = spring({
    fps,
    frame: Math.max(0, frame - 45),
    config: { stiffness: 160, damping: 16 },
  });
  const l2Op = interpolate(frame, [45, 58], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Line 3 — local 75
  const l3Op = interpolate(frame, [75, 90], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // ========== PHASE 2: CTA Button (120–200) ==========
  const phase2In = interpolate(frame, [118, 125], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const phase2Out = interpolate(frame, [190, 205], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const phase2Op = Math.min(phase2In, phase2Out);

  const btnScale = spring({
    fps,
    frame: Math.max(0, frame - 120),
    config: { stiffness: 200, damping: 14 },
  });
  const btnOp = interpolate(frame, [120, 135], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const pulse =
    frame > 144 ? 1 + 0.018 * Math.sin((frame - 144) / 9) : 1;

  const urlOp = interpolate(frame, [150, 168], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const urlY = interpolate(frame, [150, 168], [8, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // ========== PHASE 3: Outro (200–254) ==========
  const outroOp = interpolate(frame, [200, 220], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const outroScale = spring({
    fps,
    frame: Math.max(0, frame - 200),
    config: { stiffness: 160, damping: 16 },
  });
  const outroLineW = interpolate(frame, [210, 235], [0, 240], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Fade to black
  const fadeBlack = interpolate(frame, [224, 254], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill style={{ background: `${glowBlueCTA}, ${C.bg}` }}>
      <div style={gridStyle} />

      {/* PHASE 1 */}
      {frame < 122 && (
        <AbsoluteFill
          style={{
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            flexDirection: "column",
            gap: 10,
            opacity: phase1Out,
          }}
        >
          <div
            style={{
              opacity: introOp,
              fontSize: 28,
              fontWeight: 600,
              fontFamily: FONT,
              color: "#E2E8F0",
            }}
          >
            See exactly how this applies to your team.
          </div>
          <div
            style={{
              opacity: l1Op,
              transform: `scale(${l1Scale})`,
              fontSize: 52,
              fontWeight: 800,
              fontFamily: FONT,
              color: C.text1,
            }}
          >
            20–30 hours reclaimed.
          </div>
          <div
            style={{
              opacity: l2Op,
              transform: `scale(${l2Scale})`,
              fontSize: 52,
              fontWeight: 800,
              fontFamily: FONT,
              color: C.blue,
            }}
          >
            Every month.
          </div>
          <div
            style={{
              opacity: l3Op,
              fontSize: 28,
              fontWeight: 500,
              fontFamily: FONT,
              color: "#E2E8F0",
              marginTop: 8,
            }}
          >
            No new hire.{"  "}No technical skills.
          </div>
        </AbsoluteFill>
      )}

      {/* PHASE 2 */}
      {frame >= 115 && frame < 210 && (
        <AbsoluteFill
          style={{
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            flexDirection: "column",
            gap: 16,
            opacity: phase2Op,
          }}
        >
          <div
            style={{
              opacity: btnOp,
              transform: `scale(${btnScale * pulse})`,
              backgroundColor: C.blue,
              padding: "18px 44px",
              borderRadius: 12,
              boxShadow: "0 0 40px rgba(37,99,235,0.40)",
            }}
          >
            <span
              style={{
                fontSize: 20,
                fontWeight: 700,
                fontFamily: FONT,
                color: "#FFFFFF",
              }}
            >
              Request a Free Assessment →
            </span>
          </div>
          <div
            style={{
              opacity: urlOp,
              transform: `translateY(${urlY}px)`,
              fontSize: 18,
              fontWeight: 400,
              fontFamily: FONT,
              color: C.text2,
            }}
          >
            Free · No commitment · 30-minute call
          </div>
        </AbsoluteFill>
      )}

      {/* PHASE 3 */}
      {frame >= 198 && (
        <AbsoluteFill
          style={{
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            flexDirection: "column",
            opacity: outroOp,
          }}
        >
          <div
            style={{
              transform: `scale(${outroScale})`,
              fontSize: 40,
              fontWeight: 800,
              fontFamily: FONT,
              color: C.text1,
            }}
          >
            The Augmented CFO
          </div>
          <div
            style={{
              width: outroLineW,
              height: 2,
              backgroundColor: C.blue,
              borderRadius: 1,
              marginTop: 14,
            }}
          />
        </AbsoluteFill>
      )}

      {/* Fade to black */}
      <AbsoluteFill
        style={{
          backgroundColor: C.bg,
          opacity: fadeBlack,
          pointerEvents: "none",
        }}
      />
    </AbsoluteFill>
  );
};
