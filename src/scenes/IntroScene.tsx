import React from "react";
import {
  AbsoluteFill,
  useCurrentFrame,
  interpolate,
  spring,
  useVideoConfig,
} from "remotion";

export const IntroScene: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Logo / brand name animation
  const titleScale = spring({
    fps,
    frame,
    config: { damping: 100, stiffness: 200 },
  });

  const titleOpacity = interpolate(frame, [0, 20], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Subtitle appears after title
  const subtitleOpacity = interpolate(frame, [30, 50], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const subtitleY = interpolate(frame, [30, 50], [30, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Decorative line animation
  const lineWidth = interpolate(frame, [15, 45], [0, 300], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Gradient background pulse
  const bgGlow = interpolate(frame, [0, 60, 120], [0.3, 0.6, 0.3], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill
      style={{
        background: `radial-gradient(ellipse at center, rgba(37, 99, 235, ${bgGlow}) 0%, #0a0a1a 70%)`,
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        flexDirection: "column",
      }}
    >
      {/* Floating particles effect via CSS */}
      <div
        style={{
          position: "absolute",
          top: "15%",
          left: "20%",
          width: 8,
          height: 8,
          borderRadius: "50%",
          backgroundColor: "rgba(96, 165, 250, 0.4)",
          transform: `translateY(${Math.sin(frame * 0.05) * 20}px)`,
        }}
      />
      <div
        style={{
          position: "absolute",
          top: "70%",
          right: "25%",
          width: 6,
          height: 6,
          borderRadius: "50%",
          backgroundColor: "rgba(168, 85, 247, 0.4)",
          transform: `translateY(${Math.cos(frame * 0.04) * 25}px)`,
        }}
      />
      <div
        style={{
          position: "absolute",
          top: "40%",
          right: "15%",
          width: 10,
          height: 10,
          borderRadius: "50%",
          backgroundColor: "rgba(59, 130, 246, 0.3)",
          transform: `translateY(${Math.sin(frame * 0.06) * 15}px)`,
        }}
      />

      {/* Main title */}
      <div
        style={{
          opacity: titleOpacity,
          transform: `scale(${titleScale})`,
          textAlign: "center",
        }}
      >
        <div
          style={{
            fontSize: 28,
            fontFamily:
              "'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif",
            color: "rgba(148, 163, 184, 0.9)",
            letterSpacing: 8,
            textTransform: "uppercase",
            marginBottom: 20,
          }}
        >
          Introducing
        </div>
        <div
          style={{
            fontSize: 90,
            fontWeight: 800,
            fontFamily:
              "'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif",
            background:
              "linear-gradient(135deg, #ffffff 0%, #60a5fa 50%, #a78bfa 100%)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            lineHeight: 1.1,
          }}
        >
          The Augmented CFO
        </div>
      </div>

      {/* Decorative line */}
      <div
        style={{
          width: lineWidth,
          height: 3,
          background:
            "linear-gradient(90deg, transparent, #3b82f6, #8b5cf6, transparent)",
          borderRadius: 2,
          marginTop: 30,
          marginBottom: 30,
        }}
      />

      {/* Subtitle */}
      <div
        style={{
          opacity: subtitleOpacity,
          transform: `translateY(${subtitleY}px)`,
          fontSize: 32,
          fontFamily:
            "'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif",
          color: "rgba(203, 213, 225, 0.9)",
          fontWeight: 400,
          letterSpacing: 2,
        }}
      >
        AI-Powered Finance Consulting
      </div>
    </AbsoluteFill>
  );
};
