import React from "react";
import {
  AbsoluteFill,
  useCurrentFrame,
  interpolate,
  spring,
  useVideoConfig,
} from "remotion";

export const CTAScene: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const mainScale = spring({
    fps,
    frame,
    config: { damping: 100, stiffness: 180 },
  });

  const mainOpacity = interpolate(frame, [0, 15], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // CTA button animation
  const buttonOpacity = interpolate(frame, [30, 45], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const buttonY = interpolate(frame, [30, 45], [30, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Subtle button glow pulse
  const buttonGlow = interpolate(
    frame,
    [45, 60, 75, 90],
    [0.4, 0.7, 0.4, 0.7],
    {
      extrapolateLeft: "clamp",
      extrapolateRight: "clamp",
    }
  );

  // Decorative line
  const lineWidth = interpolate(frame, [10, 35], [0, 200], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Background glow
  const bgGlow = interpolate(frame, [0, 45, 90], [0.2, 0.5, 0.3], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill
      style={{
        background: `radial-gradient(ellipse at center, rgba(99, 102, 241, ${bgGlow}) 0%, #0a0a1a 70%)`,
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        alignItems: "center",
        padding: 80,
      }}
    >
      {/* Main text */}
      <div
        style={{
          opacity: mainOpacity,
          transform: `scale(${mainScale})`,
          textAlign: "center",
        }}
      >
        <div
          style={{
            fontSize: 64,
            fontWeight: 800,
            fontFamily: "sans-serif",
            color: "#ffffff",
            lineHeight: 1.2,
            marginBottom: 10,
          }}
        >
          Ready to{" "}
          <span
            style={{
              background:
                "linear-gradient(135deg, #818cf8, #60a5fa, #34d399)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
            }}
          >
            augment
          </span>
          <br />
          your finance team?
        </div>
      </div>

      {/* Decorative line */}
      <div
        style={{
          width: lineWidth,
          height: 3,
          background:
            "linear-gradient(90deg, transparent, #818cf8, #60a5fa, transparent)",
          borderRadius: 2,
          marginTop: 30,
          marginBottom: 40,
        }}
      />

      {/* CTA Button */}
      <div
        style={{
          opacity: buttonOpacity,
          transform: `translateY(${buttonY}px)`,
        }}
      >
        <div
          style={{
            background: `linear-gradient(135deg, rgba(99, 102, 241, ${buttonGlow + 0.3}), rgba(59, 130, 246, ${buttonGlow + 0.3}))`,
            borderRadius: 60,
            padding: "24px 64px",
            boxShadow: `0 0 40px rgba(99, 102, 241, ${buttonGlow * 0.5})`,
          }}
        >
          <span
            style={{
              fontSize: 32,
              fontWeight: 700,
              fontFamily: "sans-serif",
              color: "#ffffff",
              letterSpacing: 1,
            }}
          >
            Book a Free Consultation
          </span>
        </div>
      </div>

      {/* Tagline */}
      <div
        style={{
          opacity: buttonOpacity,
          marginTop: 40,
          fontSize: 22,
          fontFamily: "sans-serif",
          color: "rgba(148, 163, 184, 0.8)",
          letterSpacing: 3,
          textTransform: "uppercase",
        }}
      >
        The Augmented CFO
      </div>
    </AbsoluteFill>
  );
};
