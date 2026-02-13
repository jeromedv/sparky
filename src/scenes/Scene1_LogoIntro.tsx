import React from "react";
import {
  AbsoluteFill,
  useCurrentFrame,
  interpolate,
  spring,
  useVideoConfig,
} from "remotion";
import { COLORS, FONT, centerFlex } from "../styles";

export const Scene1_LogoIntro: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Logo fade in + slide up
  const logoOpacity = interpolate(frame, [5, 25], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const logoY = interpolate(frame, [5, 25], [40, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Subtitle fade in
  const subtitleOpacity = interpolate(frame, [20, 38], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const subtitleY = interpolate(frame, [20, 38], [20, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Blue line expands from center
  const lineWidth = interpolate(frame, [15, 45], [0, 280], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill
      style={{
        ...centerFlex,
        backgroundColor: COLORS.background,
      }}
    >
      {/* Logo */}
      <div
        style={{
          opacity: logoOpacity,
          transform: `translateY(${logoY}px)`,
          fontSize: 72,
          fontWeight: 800,
          fontFamily: FONT,
          color: COLORS.primaryText,
          letterSpacing: -1,
          textAlign: "center",
        }}
      >
        The Augmented CFO
      </div>

      {/* Blue line */}
      <div
        style={{
          width: lineWidth,
          height: 3,
          backgroundColor: COLORS.blue,
          borderRadius: 2,
          marginTop: 24,
          marginBottom: 24,
        }}
      />

      {/* Subtitle */}
      <div
        style={{
          opacity: subtitleOpacity,
          transform: `translateY(${subtitleY}px)`,
          fontSize: 28,
          fontWeight: 400,
          fontFamily: FONT,
          color: COLORS.secondaryText,
          letterSpacing: 2,
        }}
      >
        AI & Automation for Finance Teams
      </div>
    </AbsoluteFill>
  );
};
