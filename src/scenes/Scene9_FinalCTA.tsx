import React from "react";
import {
  AbsoluteFill,
  useCurrentFrame,
  interpolate,
  spring,
  useVideoConfig,
} from "remotion";
import { COLORS, FONT, centerFlex } from "../styles";

export const Scene9_FinalCTA: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // ========== PHASE 1: Impact Recap (frames 0–150) ==========

  const recapLines = [
    {
      text: "20 to 30 hours reclaimed.",
      size: 52,
      weight: 800,
      color: COLORS.primaryText,
      delay: 10,
    },
    {
      text: "Every month.",
      size: 52,
      weight: 800,
      color: COLORS.blue,
      delay: 35,
    },
    {
      text: "No new hire.",
      size: 30,
      weight: 500,
      color: COLORS.secondaryText,
      delay: 55,
    },
    {
      text: "No technical skills required.",
      size: 30,
      weight: 500,
      color: COLORS.secondaryText,
      delay: 72,
    },
  ];

  // Phase 1 fade out
  const phase1FadeOut = interpolate(frame, [130, 150], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // ========== PHASE 2: CTA Button (frames 150–420) ==========

  const phase2Frame = frame - 150;

  const ctaTextOpacity = interpolate(phase2Frame, [5, 22], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const ctaTextY = interpolate(phase2Frame, [5, 22], [20, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Button spring bounce
  const buttonScale = spring({
    fps,
    frame: Math.max(0, phase2Frame - 30),
    config: { damping: 60, stiffness: 200 },
  });
  const buttonOpacity = interpolate(phase2Frame, [28, 42], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Continuous subtle pulse on button
  const pulseScale =
    phase2Frame > 50
      ? 1 + Math.sin((phase2Frame - 50) * 0.08) * 0.015
      : 1;

  // URL fade in
  const urlOpacity = interpolate(phase2Frame, [55, 72], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Phase 2 visibility
  const phase2Opacity = interpolate(frame, [150, 155, 400, 420], [0, 1, 1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // ========== PHASE 3: Outro (frames 420–570) ==========

  const phase3Frame = frame - 420;

  const outroOpacity = interpolate(phase3Frame, [5, 25], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const outroScale = spring({
    fps,
    frame: Math.max(0, phase3Frame - 5),
    config: { damping: 120, stiffness: 180 },
  });

  // Blue line under logo
  const outroLineWidth = interpolate(phase3Frame, [15, 40], [0, 200], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Subtitle
  const outroSubOpacity = interpolate(phase3Frame, [25, 42], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Final fade to white
  const finalFade = interpolate(phase3Frame, [120, 150], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Phase 3 visibility
  const phase3Visible = frame >= 420;

  return (
    <AbsoluteFill style={{ backgroundColor: COLORS.background }}>
      {/* PHASE 1 */}
      {frame < 155 && (
        <AbsoluteFill
          style={{
            ...centerFlex,
            opacity: phase1FadeOut,
            gap: 10,
          }}
        >
          {recapLines.map((line, i) => {
            const lineScale = spring({
              fps,
              frame: Math.max(0, frame - line.delay),
              config: { damping: 80, stiffness: 180 },
            });
            const lineOpacity = interpolate(
              frame,
              [line.delay, line.delay + 12],
              [0, 1],
              { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
            );

            return (
              <div
                key={i}
                style={{
                  opacity: lineOpacity,
                  transform: `scale(${lineScale})`,
                  fontSize: line.size,
                  fontWeight: line.weight,
                  fontFamily: FONT,
                  color: line.color,
                  textAlign: "center",
                }}
              >
                {line.text}
              </div>
            );
          })}
        </AbsoluteFill>
      )}

      {/* PHASE 2 */}
      {frame >= 150 && frame < 425 && (
        <AbsoluteFill
          style={{
            ...centerFlex,
            opacity: phase2Opacity,
            gap: 12,
          }}
        >
          {/* Text */}
          <div
            style={{
              opacity: ctaTextOpacity,
              transform: `translateY(${ctaTextY}px)`,
              fontSize: 38,
              fontWeight: 500,
              fontFamily: FONT,
              color: COLORS.secondaryText,
              marginBottom: 32,
            }}
          >
            See how it applies to your team.
          </div>

          {/* Button */}
          <div
            style={{
              opacity: buttonOpacity,
              transform: `scale(${buttonScale * pulseScale})`,
              backgroundColor: COLORS.blue,
              padding: "22px 56px",
              borderRadius: 60,
              boxShadow: "0 8px 30px rgba(37, 99, 235, 0.35)",
              cursor: "default",
            }}
          >
            <span
              style={{
                fontSize: 30,
                fontWeight: 700,
                fontFamily: FONT,
                color: "#FFFFFF",
              }}
            >
              Request a Free Assessment →
            </span>
          </div>

          {/* URL */}
          <div
            style={{
              opacity: urlOpacity,
              fontSize: 22,
              fontWeight: 400,
              fontFamily: FONT,
              color: COLORS.secondaryText,
              marginTop: 24,
            }}
          >
            theaugmentedcfo.com/contact
          </div>
        </AbsoluteFill>
      )}

      {/* PHASE 3 */}
      {phase3Visible && (
        <AbsoluteFill
          style={{
            ...centerFlex,
            opacity: outroOpacity,
          }}
        >
          {/* Logo */}
          <div
            style={{
              transform: `scale(${outroScale})`,
              fontSize: 64,
              fontWeight: 800,
              fontFamily: FONT,
              color: COLORS.primaryText,
              textAlign: "center",
              letterSpacing: -1,
            }}
          >
            The Augmented CFO
          </div>

          {/* Blue line */}
          <div
            style={{
              width: outroLineWidth,
              height: 3,
              backgroundColor: COLORS.blue,
              borderRadius: 2,
              marginTop: 20,
              marginBottom: 20,
            }}
          />

          {/* Subtitle */}
          <div
            style={{
              opacity: outroSubOpacity,
              fontSize: 26,
              fontWeight: 400,
              fontFamily: FONT,
              color: COLORS.secondaryText,
              letterSpacing: 1,
            }}
          >
            AI & Automation for Finance Teams
          </div>
        </AbsoluteFill>
      )}

      {/* Final fade to white overlay */}
      <AbsoluteFill
        style={{
          backgroundColor: "#FFFFFF",
          opacity: finalFade,
          pointerEvents: "none",
        }}
      />
    </AbsoluteFill>
  );
};
