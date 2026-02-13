import React from "react";
import {
  AbsoluteFill,
  useCurrentFrame,
  interpolate,
  spring,
  useVideoConfig,
} from "remotion";
import { COLORS, FONT, centerFlex } from "../styles";

export const Scene4_SolutionReveal: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Logo appears
  const logoOpacity = interpolate(frame, [5, 22], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const logoScale = spring({
    fps,
    frame: Math.max(0, frame - 5),
    config: { damping: 120, stiffness: 180 },
  });

  // "We build AI automations" line
  const line1Opacity = interpolate(frame, [25, 40], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const line1Y = interpolate(frame, [25, 40], [20, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // "directly into your finance stack." line
  const line2Opacity = interpolate(frame, [40, 55], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const line2Y = interpolate(frame, [40, 55], [20, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Blue glow on "AI automations"
  const glowIntensity = interpolate(frame, [45, 70, 90, 110], [0, 0.6, 0.4, 0.6], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Bottom line
  const bottomOpacity = interpolate(frame, [70, 88], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const bottomY = interpolate(frame, [70, 88], [20, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill
      style={{
        ...centerFlex,
        backgroundColor: COLORS.darkBg,
        gap: 16,
      }}
    >
      {/* Logo */}
      <div
        style={{
          opacity: logoOpacity,
          transform: `scale(${logoScale})`,
          fontSize: 48,
          fontWeight: 800,
          fontFamily: FONT,
          color: "#FFFFFF",
          marginBottom: 40,
          letterSpacing: -0.5,
        }}
      >
        The Augmented CFO
      </div>

      {/* Line 1: "We build AI automations" */}
      <div
        style={{
          opacity: line1Opacity,
          transform: `translateY(${line1Y}px)`,
          fontSize: 36,
          fontWeight: 400,
          fontFamily: FONT,
          color: "#94A3B8",
          textAlign: "center",
        }}
      >
        We build{" "}
        <span
          style={{
            color: COLORS.blue,
            fontWeight: 700,
            textShadow: `0 0 ${glowIntensity * 30}px rgba(37, 99, 235, ${glowIntensity})`,
          }}
        >
          AI automations
        </span>
      </div>

      {/* Line 2: "directly into your finance stack." */}
      <div
        style={{
          opacity: line2Opacity,
          transform: `translateY(${line2Y}px)`,
          fontSize: 44,
          fontWeight: 700,
          fontFamily: FONT,
          color: "#FFFFFF",
          textAlign: "center",
        }}
      >
        directly into your finance stack.
      </div>

      {/* Bottom line */}
      <div
        style={{
          opacity: bottomOpacity,
          transform: `translateY(${bottomY}px)`,
          fontSize: 24,
          fontWeight: 400,
          fontFamily: FONT,
          color: "#64748B",
          marginTop: 40,
          textAlign: "center",
        }}
      >
        No new hires. No coding. No 6-month projects.
      </div>
    </AbsoluteFill>
  );
};
