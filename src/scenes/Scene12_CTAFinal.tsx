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
// Phase 2 — Question + CTA: local 100–200
// Phase 3 — Outro: local 200–254

export const Scene12_CTAFinal: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // ========== PHASE 1: Impact Recap (0–100) ==========
  const phase1Out = interpolate(frame, [80, 98], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Line 1 — local 10
  const l1Scale = spring({
    fps,
    frame: Math.max(0, frame - 10),
    config: { stiffness: 160, damping: 16 },
  });
  const l1Op = interpolate(frame, [10, 23], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Line 2 — local 30
  const l2Scale = spring({
    fps,
    frame: Math.max(0, frame - 30),
    config: { stiffness: 160, damping: 16 },
  });
  const l2Op = interpolate(frame, [30, 43], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Line 3 — local 50
  const l3Op = interpolate(frame, [50, 65], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // ========== PHASE 2: Question + CTA (100–200) ==========
  const phase2In = interpolate(frame, [96, 103], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const phase2Out = interpolate(frame, [185, 200], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const phase2Op = Math.min(phase2In, phase2Out);

  // Step 1: Question — frame 100
  const questionOp = interpolate(frame, [100, 120], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const questionY = interpolate(frame, [100, 120], [16, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Step 2: CTA button — frame 145 (25 frames after question starts)
  const btnScale = spring({
    fps,
    frame: Math.max(0, frame - 145),
    config: { stiffness: 200, damping: 14 },
  });
  const btnOp = interpolate(frame, [145, 160], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const pulse =
    frame > 165 ? 1 + 0.018 * Math.sin((frame - 165) / 9) : 1;

  // Step 3: Reassurance — frame 160 (15 frames after button)
  const reassureOp = interpolate(frame, [160, 175], [0, 1], {
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

      {/* PHASE 1 — Impact Recap */}
      {frame < 100 && (
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
              opacity: l1Op,
              transform: `scale(${l1Scale})`,
              fontSize: 80,
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
              fontSize: 80,
              fontWeight: 800,
              fontFamily: FONT,
              color: C.green,
            }}
          >
            Every month.
          </div>
          <div
            style={{
              opacity: l3Op,
              fontSize: 36,
              fontWeight: 500,
              fontFamily: FONT,
              color: "#CBD5E1",
              marginTop: 8,
            }}
          >
            No new hire.{"  "}No technical skills.
          </div>
        </AbsoluteFill>
      )}

      {/* PHASE 2 — Question + CTA */}
      {frame >= 95 && frame < 202 && (
        <AbsoluteFill
          style={{
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            flexDirection: "column",
            gap: 24,
            opacity: phase2Op,
          }}
        >
          {/* Step 1: Question */}
          <div
            style={{
              opacity: questionOp,
              transform: `translateY(${questionY}px)`,
              fontSize: 44,
              fontWeight: 700,
              fontFamily: FONT,
              color: C.text1,
              textAlign: "center",
            }}
          >
            What would your team do with 20 extra hours every month?
          </div>

          {/* Step 2: CTA Button */}
          <div
            style={{
              opacity: btnOp,
              transform: `scale(${btnScale * pulse})`,
              marginTop: 16,
              backgroundColor: C.blue,
              padding: "26px 64px",
              borderRadius: 12,
              boxShadow: "0 0 40px rgba(37,99,235,0.40)",
            }}
          >
            <span
              style={{
                fontSize: 32,
                fontWeight: 700,
                fontFamily: FONT,
                color: "#FFFFFF",
              }}
            >
              Request a Free Assessment →
            </span>
          </div>

          {/* Step 3: Reassurance */}
          <div
            style={{
              opacity: reassureOp,
              fontSize: 26,
              fontWeight: 500,
              fontFamily: FONT,
              color: "#94A3B8",
            }}
          >
            Free · No commitment · 30-minute call
          </div>
        </AbsoluteFill>
      )}

      {/* PHASE 3 — Outro */}
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
